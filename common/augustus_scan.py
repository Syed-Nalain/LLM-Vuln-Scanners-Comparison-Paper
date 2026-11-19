"""
augustus_scan.py - Augustus (Praetorian, v0.14.35) in the scanner comparison study (study v1, decision D64).

Augustus's own probes, each scored by the probe's own default detector. ONE probe per Augustus process (one process
per conversation goal for multi-turn probes). Augustus has no prompt cap, so whole native probes were chosen to come
close to 50 prompts per category:

  C1  50   dan.Dan_11_0, DAN_Jailbreak, DUDE, STAN, AntiDAN (1 prompt each) + promptinject.HijackHateHumans,
           HijackKillHumans, HijackLongPrompt (15 each)
  C2  50   leakreplay.LiteratureCloze 20 + leakreplay.NewsCloze 30
  C3  50   malwaregen.TopLevel 16 + grandma.Substances 10, Win10 6, Win11 6, Slurs 12
  C4  29   exploitation.JinjaTemplatePythonInjection 6 + SQLInjectionEcho 10 + SQLInjectionSystem 13 (all Augustus has)
  C6  114  lmrc.Anthropomorphisation, Profanity, QuackMedicine, SexualContent, Sexualisation, SlurUsage (14 in total)
           + realtoxicityprompts.RTPInsult 100 (the smallest whole RTP probe)
  C7  100  snowball.GraphConnectivity
  C5  10   multi-turn conversations on the 10 JailbreakBench goals of the Garak GOAT round (same goals, same order):
           goat.Goat goals 0-2, crescendo.Crescendo 3-5, tap.IterativeTAP 6-7, pair.IterativePAIR 8-9.

Target through the target proxy (temperature 0, seed 42, num_ctx 4096, max_tokens 1024), max_attempts 1, serial.
Attacker = Qwen3-Next-80B (temperature 0.7), judge (judge.Judge detector of the multi-turn probes) = GPT-OSS-120B.
Success = the probe's headline detector gives verdict 'vuln' (a probe's declared primary detector, otherwise its
only detector). A multi-turn conversation succeeds if any of its attempts is 'vuln'.

    python augustus_scan.py scan MODEL [--ports T,J,A] [--only JOB ...]   run all 36 jobs (resumable)
    python augustus_scan.py summary                                       results per model from the saved output

Output: RUNS_DIR/augustus/<model>/<job>/ (augustus_results.jsonl = every prompt, response and detector verdict;
augustus_config.yaml; proxy logs; job_result.json; DONE) and RUNS_DIR/augustus/<model>/result.json.
Needs PyYAML and the Garak GOAT round's report (for the goals).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import study_common as sc

ROOT = sc.RUNS_DIR / "augustus"
GOAT_REPORT = sc.RUNS_DIR / "garak_goat" / "gemma3-4b" / "garak.report.jsonl"

SINGLE_TURN = {
    "C1": ["dan.Dan_11_0", "dan.DAN_Jailbreak", "dan.DUDE", "dan.STAN", "dan.AntiDAN", "promptinject.HijackHateHumans",
           "promptinject.HijackKillHumans", "promptinject.HijackLongPrompt"],
    "C2": ["leakreplay.LiteratureCloze", "leakreplay.NewsCloze"],
    "C3": ["malwaregen.TopLevel", "grandma.Substances", "grandma.Win10", "grandma.Win11", "grandma.Slurs"],
    "C4": ["exploitation.JinjaTemplatePythonInjection", "exploitation.SQLInjectionEcho", "exploitation.SQLInjectionSystem"],
    "C6": ["lmrc.Anthropomorphisation", "lmrc.Profanity", "lmrc.QuackMedicine", "lmrc.SexualContent", "lmrc.Sexualisation",
           "lmrc.SlurUsage", "realtoxicityprompts.RTPInsult"],
    "C7": ["snowball.GraphConnectivity"],
}
MULTI_TURN = [("goat.Goat", 0), ("goat.Goat", 1), ("goat.Goat", 2), ("crescendo.Crescendo", 3), ("crescendo.Crescendo", 4),
              ("crescendo.Crescendo", 5), ("tap.IterativeTAP", 6), ("tap.IterativeTAP", 7), ("pair.IterativePAIR", 8),
              ("pair.IterativePAIR", 9)]
MULTI_TURN_SETTINGS = {   # TAP / PAIR would otherwise use the target model as their attacker
    "goat.Goat": {"max_turns": 5, "max_backtracks": 2},
    "crescendo.Crescendo": {"max_turns": 5, "max_backtracks": 2},
    "tap.IterativeTAP": {"width": 3, "depth": 5, "branching_factor": 2, "attacker_model": sc.ATTACKER_MODEL},
    "pair.IterativePAIR": {"n_streams": 1, "depth": 5, "attacker_model": sc.ATTACKER_MODEL},
}
# Detector that decides success, where a probe family declares a primary detector (as decision D42 for Garak).
PRIMARY_DETECTOR = {"promptinject": "promptinject.AttackRogueString", "leakreplay": "leakreplay.StartsWith",
                    "malwaregen": "malwaregen.AnyCode", "realtoxicityprompts": "toxiccomment.Toxic"}


def jobs() -> list[dict]:
    single = [{"key": probe, "cat": cat, "probe": probe, "goal_index": None}
              for cat, probes in SINGLE_TURN.items() for probe in probes]
    multi = [{"key": f"{probe}__goal{g}", "cat": "C5", "probe": probe, "goal_index": g} for probe, g in MULTI_TURN]
    return single + multi


def garak_goat_goals() -> list[str]:
    """The 10 goals of the Garak GOAT round, in Garak's seeded order, read from that run's report."""
    goals = []
    for record in sc.read_jsonl(GOAT_REPORT):
        goal = (record.get("notes") or {}).get("attack_goal") if record.get("entry_type") == "attempt" else None
        if goal and goal not in goals:
            goals.append(goal)
    assert len(goals) == 10, f"expected 10 GOAT goals, found {len(goals)}"
    return goals


def augustus_config(ports, job: dict, goal: str | None) -> dict:
    target, judge, attacker = ports
    config = {"run": {"max_attempts": 1, "timeout": "180m"},       # temperature 0 -> one attempt per prompt
              "generators": {"ollama.OllamaChat": {"host": f"http://127.0.0.1:{target}", "temperature": 0.0, "timeout": 600}},
              "judge": {"generator_type": "openai.OpenAI", "model": sc.JUDGE_MODEL,
                        "config": {"base_url": f"http://127.0.0.1:{judge}/v1", "api_key": sc.DUMMY_KEY,
                                   "temperature": sc.JUDGE_TEMPERATURE}},
              "output": {"format": "jsonl"}}
    if job["cat"] == "C5":
        config["probes"] = {"attacker_generator_type": "openai.OpenAI",
                            "attacker_config": {"model": sc.ATTACKER_MODEL, "base_url": f"http://127.0.0.1:{attacker}/v1",
                                                "api_key": sc.DUMMY_KEY, "temperature": sc.ATTACKER_TEMPERATURE},
                            "settings": {job["probe"]: {**MULTI_TURN_SETTINGS[job["probe"]], "goal": goal}}}
    return config


# =====================================================================================================================
# Results
# =====================================================================================================================

def summarize_job(folder: Path, job: dict) -> dict:
    """Success per prompt: a prompt is one (prompt text, metadata) pair; a multi-turn job is one conversation, whatever
    the number of attempt lines it wrote. Success = the headline detector flagged it (verdict not 'safe')."""
    wrapper = json.loads((folder / "wrapper_result.json").read_text(encoding="utf-8")) if (folder / "wrapper_result.json").exists() else {}
    rows = sc.read_jsonl(folder / "augustus_results.jsonl")
    detectors = sorted({r.get("detector") for r in rows})
    primary = PRIMARY_DETECTOR.get(job["probe"].split(".")[0])
    headline = primary if primary in detectors else (detectors[0] if len(detectors) == 1 else None)
    flagged = defaultdict(dict)               # prompt -> detector -> vulnerable?
    for r in rows:
        prompt_key = "conversation" if job["cat"] == "C5" else (r.get("prompt"), json.dumps(r.get("metadata"), sort_keys=True)[:500])
        vulnerable = str(r.get("verdict", "")).lower() not in ("safe", "") or str(r.get("passed")).lower() == "false"
        flagged[prompt_key][r.get("detector")] = flagged[prompt_key].get(r.get("detector"), False) or vulnerable
    summary = {"key": job["key"], "cat": job["cat"], "probe": job["probe"], "goal_index": job["goal_index"],
               "exit_code": wrapper.get("exit_code"), "duration_s": wrapper.get("duration_s"), "result_lines": len(rows),
               "prompts": len(flagged), "detectors": detectors, "headline_detector": headline or "ANY",
               "success_headline": sum((v.get(headline, False) if headline else any(v.values())) for v in flagged.values()),
               "success_any": sum(any(v.values()) for v in flagged.values()),
               "line_status": dict(Counter(r.get("status") for r in rows)),
               **{k: wrapper.get(k) for k in ("target_requests", "target_errors", "attacker_calls", "attacker_errors",
                                              "judge_calls", "judge_errors", "rejected_model_events")}}
    summary["ok"] = (wrapper.get("exit_code") == 0 and summary["prompts"] > 0 and wrapper.get("target_errors") == 0
                     and wrapper.get("rejected_model_events") == 0 and not wrapper.get("attacker_errors")
                     and not wrapper.get("judge_errors"))
    sc.write_json(folder / "job_result.json", summary)
    return summary


def summarize_model(model: str) -> dict:
    model_folder = ROOT / sc.model_dir(model)
    details, by_category = [], defaultdict(Counter)
    for job in jobs():
        folder = model_folder / job["key"]
        if not (folder / "job_result.json").exists():
            details.append({"key": job["key"], "ok": False, "missing": True})
            continue
        s = summarize_job(folder, job)
        details.append(s)
        c = by_category[job["cat"] if job["cat"] != "C5" else f"C5:{job['probe']}"]
        c["prompts"] += s["prompts"]
        c["success"] += s["success_headline"]
        c["success_any"] += s["success_any"]
        c["failed_jobs"] += not s["ok"]
    totals = {k: sum(d.get(k) or 0 for d in details)
              for k in ("target_requests", "target_errors", "attacker_calls", "judge_calls", "rejected_model_events")}
    result = {"model": model, "summarized_utc": sc.utc_now(), "jobs": len(details), "jobs_ok": sum(d.get("ok", False) for d in details),
              "by_category": {k: dict(v) for k, v in sorted(by_category.items())}, **totals, "jobs_detail": details}
    result["ok"] = result["jobs_ok"] == result["jobs"]
    sc.write_json(model_folder / "result.json", result)
    if result["ok"]:
        (model_folder / "DONE").write_text(sc.utc_now(), encoding="utf-8")
    print(f"{'OK ' if result['ok'] else 'BAD'} {model} jobs {result['jobs_ok']}/{result['jobs']}  " +
          "  ".join(f"{k}={v['success']}/{v['prompts']}" for k, v in result["by_category"].items()), flush=True)
    return result


# =====================================================================================================================
# Scan
# =====================================================================================================================

def run_job(model: str, job: dict, ports, goals: list[str]) -> str:
    import yaml
    folder = ROOT / sc.model_dir(model) / job["key"]
    if (folder / "DONE").exists():
        return "skip"
    sc.archive_incomplete(folder, label="failed")
    folder.mkdir(parents=True, exist_ok=True)
    goal = goals[job["goal_index"]] if job["goal_index"] is not None else None
    (folder / "augustus_config.yaml").write_text(yaml.safe_dump(augustus_config(ports, job, goal), sort_keys=False), encoding="utf-8")
    sc.write_json(folder / "job.json", {**job, "goal": goal})
    command = [sc.AUGUSTUS_BIN, "scan", "ollama.OllamaChat", "--config-file", str(folder / "augustus_config.yaml"),
               "--model", model, "--probe", job["probe"], "--output", str(folder / "augustus_results.jsonl"),
               "--concurrency", "1"]
    sc.run_behind_proxies("augustus", model, folder, command, ports=ports, timeout_s=4 * 3600)
    if summarize_job(folder, job)["ok"]:
        (folder / "DONE").write_text(sc.utc_now(), encoding="utf-8")
        return "ok"
    return "FAILED"           # recorded and skipped; failed jobs are re-run after the whole scan


def scan(model: str, ports, only: set[str]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    goals = garak_goat_goals()
    for job in jobs():
        if only and job["key"] not in only:
            continue
        started = sc.utc_now()
        outcome = run_job(model, job, ports, goals)
        sc.append_jsonl(ROOT / "progress.jsonl", {"model": model, "key": job["key"], "started_utc": started,
                                                  "finished_utc": sc.utc_now(), "outcome": outcome})
        print(f"{model} {job['key']}: {outcome}", flush=True)
    summarize_model(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["scan", "summary"])
    parser.add_argument("model", nargs="?")
    parser.add_argument("--ports", default=None, help="target,judge,attacker ports (default 11500,11600,11601)")
    parser.add_argument("--only", nargs="*", default=[], help="scan: run only these job keys")
    a = parser.parse_args()
    if a.command == "scan":
        assert a.model in sc.STUDY_MODELS, f"unknown model {a.model}"
        scan(a.model, sc.parse_ports(a.ports), set(a.only))
    else:
        for model in sc.STUDY_MODELS:
            if (ROOT / sc.model_dir(model)).exists():
                summarize_model(model)


if __name__ == "__main__":
    main()
