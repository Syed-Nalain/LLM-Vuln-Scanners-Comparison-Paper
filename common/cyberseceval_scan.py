"""
cyberseceval_scan.py - Meta CyberSecEval (PurpleLlama CybersecurityBenchmarks) in the scanner comparison study.

CyberSecEval is single-turn only. Its Insecure Code Detector needs semgrep, which has no Windows build, so the
benchmarks run inside WSL2 (Ubuntu 22.04, Python venv ~/.venvs/cse) with WSL "mirrored" networking, so that WSL
reaches the Windows-side study proxies at 127.0.0.1 (decision D33).

Prompts: seeded subsets of CyberSecEval's own datasets (seed 42, sampled WITHIN each stratum), written in
CyberSecEval's own file format so the scanner runs them natively. Same 200 prompts for every model:

    C1  prompt_injection   50 of 251   proportional over the 15 injection variants (largest remainder, each >= 1)
    C3  mitre              50 of 1000  5 per MITRE ATT&CK category (10 categories)
    C4  instruct           25 of 1916  3 per language (8 languages) + 1 extra from one seeded language
    C4  autocomplete       25 of 1916  same rule, drawn independently
    --  mitre_frr          50 of 750   false-refusal control (benign prompts), simple random sample

Native verdicts: prompt injection - the judge answers each case's judge question; MITRE - local refusal check,
then expansion + judge (malicious / benign); instruct / autocomplete - the Insecure Code Detector (no LLM);
FRR - local refusal check. Judge and expansion LLM = GPT-OSS-120B through the judge proxy. No attacker LLM.

    python cyberseceval_scan.py subsets                         build the seeded prompt subsets (no model calls)
    python cyberseceval_scan.py scan MODEL [--ports T,J,A]      run the 5 benchmarks for one model (from Windows)
    python cyberseceval_scan.py summary                         results per model from CyberSecEval's stats files

Output: RUNS_DIR/cyberseceval/subsets/ and RUNS_DIR/cyberseceval/<model>/<benchmark>/ (responses.json, judge.json,
stats.json, stdout/stderr, DONE) plus the proxy logs in RUNS_DIR/cyberseceval/<model>/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

import study_common as sc

ROOT = sc.RUNS_DIR / "cyberseceval"
SUBSET_DIR = ROOT / "subsets"
DATASETS = Path(sc.CYBERSECEVAL_REPO) / "CybersecurityBenchmarks" / "datasets"
WSL_DISTRO = os.environ.get("CSE_WSL_DISTRO", "Ubuntu-22.04")
WSL_VENV_PYTHON = os.environ.get("CSE_WSL_PYTHON", "~/.venvs/cse/bin/python")

# benchmark folder / subset name -> (CyberSecEval benchmark name, needs the judge LLM, needs the expansion LLM)
BENCHMARKS = {"prompt_injection": ("prompt-injection", True, False),
              "mitre": ("mitre", True, True),
              "mitre_frr": ("mitre-frr", False, False),
              "instruct": ("instruct", False, False),
              "autocomplete": ("autocomplete", False, False)}


# =====================================================================================================================
# 1. Seeded prompt subsets
# =====================================================================================================================

def group_by(rows: list[dict], key: str) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row[key])].append(i)
    return dict(sorted(groups.items()))


def sample_proportional(rows, key, n, rng) -> list[int]:
    """n rows spread over the strata in proportion to stratum size (largest remainder; every stratum >= 1)."""
    groups = group_by(rows, key)
    share = {k: n * len(v) / len(rows) for k, v in groups.items()}
    alloc = {k: max(1, int(x)) for k, x in share.items()}
    for k in sorted(share, key=lambda k: (-(share[k] - int(share[k])), k)):
        if sum(alloc.values()) >= n:
            break
        alloc[k] += 1
    return sorted(i for k, idx in groups.items() for i in rng.sample(idx, alloc[k]))


def sample_per_stratum(rows, key, each, n, rng) -> list[int]:
    """`each` rows from every stratum, then the remaining n - each*strata rows from distinct seeded strata."""
    groups = group_by(rows, key)
    picked = {k: rng.sample(idx, each) for k, idx in groups.items()}
    for k in rng.sample(sorted(groups), n - each * len(groups)):
        picked[k].append(rng.choice([i for i in groups[k] if i not in picked[k]]))
    return sorted(i for idx in picked.values() for i in idx)


SUBSET_PLAN = [   # name, dataset file, stratum key, sampler
    ("prompt_injection", "prompt_injection/prompt_injection.json", "injection_variant",
     lambda rows, rng: sample_proportional(rows, "injection_variant", 50, rng)),
    ("mitre", "mitre/mitre_benchmark_100_per_category_with_augmentation.json", "mitre_category",
     lambda rows, rng: sample_per_stratum(rows, "mitre_category", 5, 50, rng)),
    ("instruct", "instruct/instruct.json", "language",
     lambda rows, rng: sample_per_stratum(rows, "language", 3, 25, rng)),
    ("autocomplete", "autocomplete/autocomplete.json", "language",
     lambda rows, rng: sample_per_stratum(rows, "language", 3, 25, rng)),
    ("mitre_frr", "mitre_frr/mitre_frr.json", None,
     lambda rows, rng: sorted(rng.sample(range(len(rows)), 50))),
]


def build_subsets() -> None:
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": sc.SEED, "source_dir": str(DATASETS), "subsets": {}}
    for name, relative_path, key, sampler in SUBSET_PLAN:
        rows = json.loads((DATASETS / relative_path).read_text(encoding="utf-8"))
        indices = sampler(rows, random.Random(sc.SEED))          # fresh seeded generator per benchmark
        subset = [rows[i] for i in indices]
        (SUBSET_DIR / f"{name}.json").write_text(json.dumps(subset, indent=1, ensure_ascii=False), encoding="utf-8")
        per_stratum = dict(sorted(Counter(str(r[key]) for r in subset).items())) if key else None
        manifest["subsets"][name] = {"source": relative_path, "pool": len(rows), "n": len(subset), "stratum_key": key,
                                     "per_stratum": per_stratum, "source_indices": indices}
        print(f"{name}: {len(rows)} -> {len(subset)} {per_stratum or ''}")
    (SUBSET_DIR / "subsets_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


# =====================================================================================================================
# 2. Scan (Windows side starts the proxies; the benchmarks run inside WSL)
# =====================================================================================================================

def wsl_path(path) -> str:
    """D:\\project\\x -> /mnt/d/project/x"""
    p = PureWindowsPath(str(path))
    return f"/mnt/{p.drive[0].lower()}/" + "/".join(p.parts[1:]) if p.drive else str(path)


def scan(model: str, ports) -> None:
    folder = ROOT / sc.model_dir(model)
    folder.mkdir(parents=True, exist_ok=True)
    worker = ("cd " + json.dumps(wsl_path(Path(__file__).resolve().parent)) + " && " + WSL_VENV_PYTHON +
              " cyberseceval_scan.py wsl-worker " + " ".join(json.dumps(x) for x in (
                  model, wsl_path(folder), wsl_path(SUBSET_DIR), wsl_path(sc.CYBERSECEVAL_REPO), str(ports[0]), str(ports[1]),
                  str(sc.SMOKE_LIMIT))))
    wrapper = sc.run_behind_proxies("cyberseceval", model, folder, ["wsl.exe", "-d", WSL_DISTRO, "--", "bash", "-c", worker],
                                    ports=ports, timeout_s=4 * 3600)
    print(json.dumps(wrapper, indent=1))


def wsl_worker(model, out_dir, subset_dir, repo, target_port, judge_port, smoke_limit="0") -> None:
    """Runs INSIDE WSL with the CyberSecEval venv: one CyberSecEval process per benchmark. A benchmark with a DONE file
    is skipped (resumable); a failed benchmark is recorded and the next one still runs."""
    target = f"OPENAI::{model}::ollama::http://127.0.0.1:{target_port}/v1"          # key field is unused by Ollama
    judge = f"OPENAI::{sc.JUDGE_MODEL}::{sc.DUMMY_KEY}::http://127.0.0.1:{judge_port}/v1"  # proxy injects the real key
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "OPENAI_BASE_URL")}
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
    for name, (benchmark, uses_judge, uses_expansion) in BENCHMARKS.items():
        folder = Path(out_dir) / name
        if (folder / "DONE").exists():
            print(f"skip {name} (DONE)", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "CybersecurityBenchmarks.benchmark.run", f"--benchmark={benchmark}",
                   f"--prompt-path={Path(subset_dir) / (name + '.json')}", f"--response-path={folder / 'responses.json'}",
                   f"--stat-path={folder / 'stats.json'}", f"--llm-under-test={target}"]
        if uses_judge:
            command += [f"--judge-llm={judge}", f"--judge-response-path={folder / 'judge.json'}"]
        if uses_expansion:
            command += [f"--expansion-llm={judge}"]
        if int(smoke_limit):                                              # smoke tests only
            command += [f"--num-test-cases={smoke_limit}"]
        t0 = time.time()
        with (folder / "stdout.txt").open("w") as out, (folder / "stderr.txt").open("w") as err:
            exit_code = subprocess.run(command, cwd=repo, env=env, stdout=out, stderr=err).returncode
        if exit_code == 0 and (folder / "stats.json").exists() and (folder / "stats.json").stat().st_size:
            (folder / "DONE").touch()
        status = {"benchmark": name, "exit": exit_code, "seconds": round(time.time() - t0), "at": sc.utc_now()}
        sc.append_jsonl(Path(out_dir) / "cse_status.jsonl", status)
        print(json.dumps(status), flush=True)


# =====================================================================================================================
# 3. Summary (CyberSecEval's own stats.json per benchmark)
# =====================================================================================================================

def model_summary(folder: Path) -> dict:
    def stats(name):
        return list(json.loads((folder / name / "stats.json").read_text(encoding="utf-8")).values())[0]

    out = {}
    pi = stats("prompt_injection")
    total = pi["stat_per_model"]
    out["C1_prompt_injection"] = {
        "success": total["injection_successful_count"],
        "total": total["injection_successful_count"] + total["injection_unsuccessful_count"],
        "by_variant": {k: [v["injection_successful_count"], v["total_count"]]
                       for k, v in pi["stat_per_model_per_injection_variant"].items()},
        "by_type": {k: [v["injection_successful_count"], v["total_count"]]
                    for k, v in pi["stat_per_model_per_injection_type"].items()}}
    mitre = stats("mitre")
    out["C3_mitre"] = {"success": sum(v["malicious_count"] for v in mitre.values()),
                       "refusal": sum(v["refusal_count"] for v in mitre.values()),
                       "benign": sum(v["benign_count"] for v in mitre.values()),
                       "total": sum(v["total_count"] for v in mitre.values()),
                       "by_category": {k: [v["malicious_count"], v["refusal_count"], v["total_count"]]
                                       for k, v in mitre.items()}}
    for name in ("instruct", "autocomplete"):
        s = stats(name)
        out[f"C4_{name}"] = {"success": sum(v["vulnerable_suggestion_count"] for v in s.values()),
                             "total": sum(v["total_count"] for v in s.values()),
                             "by_language": {k: [v["vulnerable_suggestion_count"], v["total_count"]] for k, v in s.items()},
                             "mean_bleu": round(sum(v["bleu"] for v in s.values()) / len(s), 2)}
    frr = stats("mitre_frr")
    out["control_frr"] = {"refusal": frr["refusal_count"], "total": frr["refusal_count"] + frr["accept_count"]}
    attacks = [k for k in out if k != "control_frr"]
    out["attack_total"] = {"success": sum(out[k]["success"] for k in attacks), "total": sum(out[k]["total"] for k in attacks)}
    wrapper = json.loads((folder / "wrapper_result.json").read_text(encoding="utf-8"))
    out["requests"] = {k: wrapper[k] for k in ("target_requests", "target_errors", "judge_calls", "judge_errors",
                                               "rejected_model_events", "duration_s")}
    out["benchmarks_done"] = sorted(p.parent.name for p in folder.glob("*/DONE"))
    out["status_rows"] = sc.read_jsonl(folder / "cse_status.jsonl")
    return out


def summary() -> dict:
    result = {sc.model_dir(m): model_summary(ROOT / sc.model_dir(m)) for m in sc.STUDY_MODELS
              if (ROOT / sc.model_dir(m) / "wrapper_result.json").exists()}
    sc.write_json(ROOT / "summary.json", result)
    for model, r in result.items():
        print(f"{model:32s} attacks {r['attack_total']['success']}/{r['attack_total']['total']}  "
              f"false refusals {r['control_frr']['refusal']}/{r['control_frr']['total']}")
    return result


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "wsl-worker":          # internal: runs inside WSL
        return wsl_worker(*sys.argv[2:9])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["subsets", "scan", "summary"])
    parser.add_argument("model", nargs="?")
    parser.add_argument("--ports", default=None, help="target,judge,attacker ports (default 11500,11600,11601)")
    a = parser.parse_args()
    if a.command == "subsets":
        build_subsets()
    elif a.command == "scan":
        assert a.model in sc.STUDY_MODELS, f"unknown model {a.model}"
        scan(a.model, sc.parse_ports(a.ports))
    else:
        summary()


if __name__ == "__main__":
    main()
