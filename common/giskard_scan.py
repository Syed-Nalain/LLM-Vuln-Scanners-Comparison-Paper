"""
giskard_scan.py - Giskard (giskard-scan 3.x) in the scanner comparison study (study v1, decision D63).

Giskard's own scenario generators are used through `generate_suite` and `Scenario.run`, the functions Giskard's
`vulnerability_scan` calls. The suites are generated ONCE (seed 42) and the same scenarios are run against every model.

  SINGLE-TURN  AdversarialScenarioGenerator + PromptInjectionScenarioGenerator, target_mode 'singleturn'.
               max_scenarios 400 is enough to reach Giskard's offline ceiling: 8 adversarial categories x their cap
               of 10 + all 18 prompt-injection techniques = 98 scenarios.
  MULTI-TURN   CrescendoAttackScenarioGenerator + GOATAttackScenarioGenerator, max_turns 5 (Giskard default 10),
               10 scenarios, target_mode 'multiturn'.
The 3 Hugging Face generators (do-not-answer, harmbench, GCG) produce nothing offline (no downloads were allowed),
so they are left out.

Generator / attacker LLM = Qwen3-Next-80B (temperature 0.7), judge = GPT-OSS-120B (temperature 0), both through the
study brain proxies; the target goes through the target proxy (study_common.py).
Success = Giskard status FAIL (the check failed = vulnerability found). PASS = safe.

    python giskard_scan.py generate                   build the two suites once (stub target, no target calls)
    python giskard_scan.py scan MODEL [--ports T,J,A] run both suites against one model
    python giskard_scan.py summary                    results per model from the saved results.jsonl files

`generate` and `scan` can be started with any Python; Giskard itself runs in GISKARD_PYTHON.
Output: RUNS_DIR/giskard/generate/suite_single.json, suite_multi.json and RUNS_DIR/giskard/<model>/results.jsonl
(one line per scenario with Giskard's full trace and check messages), proxy logs, result.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import study_common as sc

ROOT = sc.RUNS_DIR / "giskard"
SUITE_DIR = ROOT / "generate"
THIS_FILE = str(Path(__file__).resolve())
N_SINGLE, N_MULTI, MAX_TURNS = 400, 10, 5
os.environ.setdefault("GISKARD_TELEMETRY_DISABLED", "1")       # must be set before Giskard is imported


# =====================================================================================================================
# 1. Giskard wiring (runs inside Giskard's Python)
# =====================================================================================================================

def configure_giskard(judge_port: int, attacker_port: int) -> None:
    """Named giskard-llm providers 'attacker' and 'judge' pointing at our brain proxies: the default generator (scenario
    generation and adversarial turns) uses the attacker, the default judge (all LLM checks) uses the judge."""
    from giskard.agents import Generator
    from giskard.checks import set_default_generator, set_default_judge
    from giskard.llm import configure
    configure("attacker", provider="openai", base_url=f"http://127.0.0.1:{attacker_port}/v1", api_key=sc.DUMMY_KEY)
    configure("judge", provider="openai", base_url=f"http://127.0.0.1:{judge_port}/v1", api_key=sc.DUMMY_KEY)
    set_default_generator(Generator(model=f"attacker/{sc.ATTACKER_MODEL}", params={"temperature": sc.ATTACKER_TEMPERATURE}))
    set_default_judge(Generator(model=f"judge/{sc.JUDGE_MODEL}", params={"temperature": sc.JUDGE_TEMPERATURE}))


def call_target(target_port: int, model: str, messages: list[dict]) -> str:
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{target_port}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as r:
        return json.loads(r.read())["message"]["content"]


def trace_to_messages(current_input: str, trace) -> list[dict]:
    """Earlier turns from Giskard's trace (inputs -> user, outputs -> assistant), then the current input. The current
    input is not appended twice if the trace already ends with it (same guard as DeepTeam's D56 fix)."""
    messages = []
    for interaction in (getattr(trace, "interactions", None) or []):
        if getattr(interaction, "inputs", None) is not None:
            messages.append({"role": "user", "content": str(interaction.inputs)})
        if getattr(interaction, "outputs", None) is not None:
            messages.append({"role": "assistant", "content": str(interaction.outputs)})
    if not (messages and messages[-1]["role"] == "user" and messages[-1]["content"] == current_input):
        messages.append({"role": "user", "content": current_input})
    return messages


def single_turn_target(target_port: int, model: str):
    async def target(inputs: str) -> str:
        return await asyncio.to_thread(call_target, target_port, model, [{"role": "user", "content": inputs}])
    return target


def multi_turn_target(target_port: int, model: str):
    async def target(inputs: str, trace=None) -> str:            # Giskard passes the conversation so far as `trace`
        return await asyncio.to_thread(call_target, target_port, model, trace_to_messages(inputs, trace))
    return target


async def generate_worker(out_dir: Path) -> None:
    from giskard.checks import Suite
    from giskard.scan.catalog import generate_suite
    from giskard.scan.generators.adversarial import AdversarialScenarioGenerator
    from giskard.scan.generators.crescendo import CrescendoAttackScenarioGenerator
    from giskard.scan.generators.goat import GOATAttackScenarioGenerator
    from giskard.scan.generators.prompt_injection import PromptInjectionScenarioGenerator
    plans = {"single": ([AdversarialScenarioGenerator(), PromptInjectionScenarioGenerator()], N_SINGLE, "singleturn"),
             "multi": ([CrescendoAttackScenarioGenerator(max_turns=MAX_TURNS), GOATAttackScenarioGenerator(max_turns=MAX_TURNS)],
                       N_MULTI, "multiturn")}
    summary = {}
    for name, (generators, n, mode) in plans.items():
        suite_file = out_dir / f"suite_{name}.json"
        if suite_file.exists():
            summary[name] = "exists"
            continue
        suite = await generate_suite(sc.NEUTRAL_PURPOSE, ["en"], generators, max_scenarios=n, seed=sc.SEED, target_mode=mode)
        text = suite.model_dump_json(indent=1)
        Suite.model_validate_json(text)                      # must reload, or the scan step could not reuse it
        suite_file.write_text(text, encoding="utf-8")
        summary[name] = {"scenarios": len(suite.scenarios), "target_mode": mode,
                         "generators": [type(g).__name__ for g in generators], "names": [s.name for s in suite.scenarios]}
        print(name, len(suite.scenarios), flush=True)
    (out_dir / "generate_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


async def scan_worker(model: str, out_dir: Path, suite_dir: Path, target_port: int) -> None:
    """Runs every scenario one at a time; each result is appended to results.jsonl as soon as it finishes, so a crash
    loses at most one scenario and finished scenarios are skipped on restart."""
    from giskard.checks import Suite
    import giskard.scan              # noqa: F401  registers the 'giskard.scan' prompt templates the scenarios use
    import giskard.scan.generators   # noqa: F401
    results_file = out_dir / "results.jsonl"
    done = {r["key"] for r in sc.read_jsonl(results_file)}
    for part, mode in (("single", "singleturn"), ("multi", "multiturn")):
        suite = Suite.model_validate_json((suite_dir / f"suite_{part}.json").read_text(encoding="utf-8"))
        target = multi_turn_target(target_port, model) if mode == "multiturn" else single_turn_target(target_port, model)
        scenarios = suite.scenarios[:sc.SMOKE_LIMIT] if sc.SMOKE_LIMIT else suite.scenarios   # smoke tests only
        for i, scenario in enumerate(scenarios):
            key = f"{part}_{i:03d}"
            if key in done:
                continue
            t0 = time.time()
            record = {"key": key, "part": part, "index": i, "scenario": scenario.name, "tags": getattr(scenario, "tags", None)}
            try:
                result = await scenario.run(target=target, return_exception=True)
                record.update({"ok": True, "status": str(getattr(result.status, "value", result.status)),
                               "result": json.loads(result.model_dump_json())})
            except Exception as e:  # noqa: BLE001 - recorded and skipped; re-run after the scan
                record.update({"ok": False, "status": "harness_error", "error": repr(e)[:2000],
                               "trace": traceback.format_exc()[-4000:]})
            record["duration_s"] = round(time.time() - t0, 1)
            sc.append_jsonl(results_file, record)
            print(f"[{part} {i + 1}/{len(scenarios)}] {scenario.name[:60]} -> {record['status']}", flush=True)
    print("GISKARD_DONE", flush=True)


# =====================================================================================================================
# 2. Launcher and summary (any Python)
# =====================================================================================================================

def group_of(record: dict) -> str:
    """Generator group used in the results: a 'generator' tag if Giskard set one, else the scenario name prefix."""
    tags = record.get("tags") or []
    return next((t for t in tags if isinstance(t, str) and "generator" in t), None) or record["scenario"].split(":")[0][:40]


def summarize(folder: Path) -> dict:
    rows = sc.read_jsonl(folder / "results.jsonl")
    groups = defaultdict(Counter)
    for r in rows:
        key = f"{r['part']}|{group_of(r)}"
        groups[key]["scenarios"] += 1
        groups[key][r["status"]] += 1
    status = Counter(r["status"] for r in rows)
    expected = sum(v["scenarios"] for v in json.loads((SUITE_DIR / "generate_summary.json").read_text(encoding="utf-8")).values()
                   if isinstance(v, dict))
    if sc.SMOKE_LIMIT:                                                   # smoke tests only
        expected = 2 * sc.SMOKE_LIMIT
    return {"scenarios": len(rows), "expected": expected, "status": dict(status), "attack_success": status.get("fail", 0),
            "errors": status.get("error", 0) + status.get("harness_error", 0),
            "by_part_generator": {k: dict(v) for k, v in sorted(groups.items())}}


def generate() -> None:
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    _, judge_port, attacker_port = sc.MAIN_PORTS
    sc.run_behind_proxies("giskard", "gemma3:4b", SUITE_DIR, [sc.GISKARD_PYTHON, THIS_FILE, "generate-worker", str(SUITE_DIR),
                                                              str(judge_port), str(attacker_port)],
                          stub_mode="refuse", timeout_s=6 * 3600)


def scan(model: str, ports) -> None:
    folder = ROOT / sc.model_dir(model)
    if (folder / "DONE").exists():
        print("skip (done)", model)
        return
    folder.mkdir(parents=True, exist_ok=True)          # results.jsonl is resumable, so nothing is archived
    started = sc.utc_now()
    command = [sc.GISKARD_PYTHON, THIS_FILE, "scan-worker", model, str(folder), str(SUITE_DIR)] + [str(p) for p in ports]
    wrapper = sc.run_behind_proxies("giskard", model, folder, command, ports=ports, timeout_s=20 * 3600)
    result = {"model": model, "started_utc": started, "finished_utc": sc.utc_now(), "wrapper_exit": wrapper["exit_code"],
              "duration_s": wrapper["duration_s"], **summarize(folder),
              **{k: wrapper[k] for k in ("target_requests", "target_errors", "attacker_calls", "attacker_errors",
                                         "judge_calls", "judge_errors", "rejected_model_events")}}
    result["ok"] = (result["scenarios"] == result["expected"] and result["status"].get("harness_error", 0) == 0
                    and result["target_errors"] == 0 and result["rejected_model_events"] == 0)
    sc.write_json(folder / "result.json", result)
    sc.append_jsonl(ROOT / "progress.jsonl", result)
    if result["ok"]:
        (folder / "DONE").write_text(sc.utc_now(), encoding="utf-8")
    print(("OK  " if result["ok"] else "BAD ") + f"{model} scenarios={result['scenarios']} status={result['status']}")


def summary() -> dict:
    out = {sc.model_dir(m): summarize(ROOT / sc.model_dir(m)) for m in sc.STUDY_MODELS
           if (ROOT / sc.model_dir(m) / "results.jsonl").exists()}
    sc.write_json(ROOT / "summary.json", out)
    for model, s in out.items():
        print(f"{model:32s} FAIL (attack succeeded) {s['attack_success']}/{s['scenarios']}")
    return out


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "generate-worker":       # internal: runs in Giskard's Python
        out_dir, judge_port, attacker_port = Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        configure_giskard(judge_port, attacker_port)
        return asyncio.run(generate_worker(out_dir))
    if len(sys.argv) > 1 and sys.argv[1] == "scan-worker":           # internal: runs in Giskard's Python
        model, out_dir, suite_dir = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
        target_port, judge_port, attacker_port = (int(x) for x in sys.argv[5:8])
        configure_giskard(judge_port, attacker_port)
        return asyncio.run(scan_worker(model, out_dir, suite_dir, target_port))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "scan", "summary"])
    parser.add_argument("model", nargs="?")
    parser.add_argument("--ports", default=None, help="target,judge,attacker ports (default 11500,11600,11601)")
    a = parser.parse_args()
    if a.command == "generate":
        generate()
    elif a.command == "scan":
        assert a.model in sc.STUDY_MODELS, f"unknown model {a.model}"
        scan(a.model, sc.parse_ports(a.ports))
    else:
        summary()


if __name__ == "__main__":
    main()
