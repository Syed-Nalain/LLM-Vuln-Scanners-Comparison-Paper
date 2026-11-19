"""
promptfoo_scan.py - Promptfoo red teaming in the scanner comparison study (study v1, decisions D57-D61).

Promptfoo runs in its FULL REMOTE mode (D59): Promptfoo's hosted model writes the attacks, drives the multi-turn
strategies and grades the replies, so its attack-success numbers come from Promptfoo's own grader. Only the target
model is ours, reached through the study target proxy (study_common.py), which logs every prompt and response.

Plan (D60): 14 plugins x 3 base tests = 42 base tests.
  SINGLE-TURN  strategies basic, prompt-injection (jailbreak templates), base64, rot13, leetspeak on every base test.
  MULTI-TURN   10 conversations: crescendo 3, goat 3, jailbreak (meta) 2, jailbreak:tree 2, each capped at 5 turns /
               iterations / depth. Promptfoo's own strategy `numTests` takes the FIRST N applicable tests, which put
               all 10 on one plugin, so a seeded spread is used instead (D61, see select_multi_turn).
The tests are generated ONCE and the same file is run against every model.

Success = Promptfoo's failureReason 1 (the grader's assertion failed = attack succeeded); 0 = safe; 2 = error.

Run with a Python that has PyYAML; Promptfoo itself runs with Node (PROMPTFOO_ENTRY):

    python promptfoo_scan.py generate                   generate the tests once (stub target, no model calls)
    python promptfoo_scan.py select-multiturn           apply the D61 seeded multi-turn selection (no model calls)
    python promptfoo_scan.py eval MODEL [--ports T,J,A] run the shared tests against one model
    python promptfoo_scan.py summary                    results per model from Promptfoo's saved output

Output: RUNS_DIR/promptfoo/generate/ (redteam.yaml = the shared tests) and RUNS_DIR/promptfoo/<model>/
(promptfoo_results.json = every test with prompt, response and grade; proxy logs; result.json).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import study_common as sc

ROOT = sc.RUNS_DIR / "promptfoo"
GENERATE_DIR = ROOT / "generate"
EMPTY_CODEX_HOME = sc.CODE_DIR / "empty_codex_home"

PLUGINS = ["hijacking", "system-prompt-override", "pii:direct", "prompt-extraction", "harmful:cybercrime",
           "harmful:cybercrime:malicious-code", "harmful:violent-crime", "harmful:illegal-activities", "harmful:hate",
           "harmful:insults", "bias:gender", "bias:race", "harmful:misinformation-disinformation", "hallucination"]
STRATEGIES = ["basic", "prompt-injection", "base64", "rot13", "leetspeak",
              {"id": "crescendo", "config": {"numTests": 3, "maxTurns": 5, "maxBacktracks": 2}},
              {"id": "goat", "config": {"numTests": 3, "maxTurns": 5}},
              {"id": "jailbreak", "config": {"numTests": 2, "numIterations": 5}},
              {"id": "jailbreak:tree", "config": {"numTests": 2, "maxDepth": 5}}]
MULTI_TURN = ("crescendo", "goat", "jailbreak:meta", "jailbreak:tree")
MULTI_TURN_PLAN = ["crescendo"] * 3 + ["goat"] * 3 + ["jailbreak:meta"] * 2 + ["jailbreak:tree"] * 2


def load_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_yaml(path: Path, obj) -> None:
    import yaml
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def promptfoo_config(model: str) -> dict:
    return {"description": f"study_v1 promptfoo - {model}", "prompts": ["{{prompt}}"],
            "targets": [{"id": f"ollama:chat:{model}", "label": "target"}],
            "redteam": {"purpose": sc.NEUTRAL_PURPOSE, "numTests": 3, "plugins": PLUGINS, "strategies": STRATEGIES}}


def run_promptfoo(model: str, folder: Path, args: list[str], ports=sc.MAIN_PORTS, stub_mode=None) -> dict:
    """Run one Promptfoo command behind the proxies, in full remote mode (D57/D59). Promptfoo would otherwise use a
    local Codex/ChatGPT login as its default OpenAI provider, so CODEX_HOME points at an empty folder."""
    EMPTY_CODEX_HOME.mkdir(parents=True, exist_ok=True)
    return sc.run_behind_proxies("promptfoo", model, folder, ["node", sc.PROMPTFOO_ENTRY] + args, ports=ports,
                                 stub_mode=stub_mode, timeout_s=20 * 3600,
                                 env_update={"CODEX_HOME": str(EMPTY_CODEX_HOME)},
                                 env_remove=("PROMPTFOO_DISABLE_REMOTE_GENERATION", "CODEX_API_KEY"))


def strategy_of(test: dict) -> str | None:
    return (test.get("metadata") or {}).get("strategyId")


# =====================================================================================================================
# Steps
# =====================================================================================================================

def generate() -> None:
    """Generate the attack tests once. Generation needs no target model; the stub sits on the target port."""
    folder = GENERATE_DIR
    if (folder / "DONE").exists():
        print("skip (done) generate")
        return
    folder.mkdir(parents=True, exist_ok=True)
    save_yaml(folder / "promptfooconfig.yaml", promptfoo_config("gemma3:4b"))
    run_promptfoo("gemma3:4b", folder, ["redteam", "generate", "-c", str(folder / "promptfooconfig.yaml"),
                                        "-o", str(folder / "redteam.yaml"), "--force", "--no-cache", "--no-progress-bar"],
                  stub_mode="refuse")
    tests = (load_yaml(folder / "redteam.yaml") or {}).get("tests") or []
    summary = {"finished_utc": sc.utc_now(), "tests": len(tests),
               "by_strategy": dict(Counter(strategy_of(t) or "basic" for t in tests)),
               "by_plugin": dict(Counter((t.get("metadata") or {}).get("pluginId") for t in tests))}
    sc.write_json(folder / "generate_summary.json", summary)
    print(json.dumps(summary, indent=1))
    if tests:
        (folder / "DONE").write_text(sc.utc_now(), encoding="utf-8")


def select_multi_turn(native_file: Path, out_file: Path, record_file: Path) -> list[dict]:
    """D61 seeded spread. Promptfoo's first-N rule put all 10 multi-turn conversations on one plugin. Instead:
    10 different plugins are sampled with seed 42 (system-prompt-override excluded: its chat-format tests get no
    strategies natively), one base test of each plugin is drawn with the same generator, and it is wrapped with the
    exact provider / metadata / metric of Promptfoo's own generated test for that strategy. Single-turn tests are
    left unchanged."""
    data = load_yaml(native_file)
    tests = data["tests"]
    template = {}                                       # first Promptfoo-generated test of each multi-turn strategy
    for t in tests:
        if strategy_of(t) in MULTI_TURN and strategy_of(t) not in template:
            template[strategy_of(t)] = t
    base_tests = [t for t in tests if not strategy_of(t)]
    plugins = sorted({t["metadata"]["pluginId"] for t in base_tests} - {"system-prompt-override"})
    rng = random.Random(sc.SEED)
    chosen_plugins = rng.sample(plugins, 10)
    new_tests, record = [], []
    for plugin, strategy in zip(chosen_plugins, MULTI_TURN_PLAN):
        candidates = [t for t in base_tests if t["metadata"]["pluginId"] == plugin]
        base = candidates[rng.randrange(len(candidates))]
        tmpl, test = template[strategy], copy.deepcopy(base)
        metric_suffix = tmpl["assert"][0]["metric"].split("/", 1)[1]
        for assertion in test.get("assert") or []:
            assertion["metric"] = f"{assertion['metric']}/{metric_suffix}"
        test["metadata"].update({"strategyId": strategy, "originalText": base["vars"]["prompt"],
                                 "strategyConfig": copy.deepcopy(tmpl["metadata"]["strategyConfig"])})
        test["provider"] = copy.deepcopy(tmpl["provider"])
        new_tests.append(test)
        record.append({"strategy": strategy, "plugin": plugin, "base_test_index": candidates.index(base)})
    data["tests"] = [t for t in tests if strategy_of(t) not in MULTI_TURN] + new_tests
    save_yaml(out_file, data)
    record_file.write_text(json.dumps({"decision": "D61 seeded spread (researcher 2026-11-10)", "seed": sc.SEED,
                                       "eligible_plugins": plugins, "selection": record,
                                       "templates_from": f"Promptfoo-generated multi-turn tests in {native_file.name}"},
                                      indent=2), encoding="utf-8")
    print(len(data["tests"]), "tests:", dict(Counter(strategy_of(t) or "basic" for t in data["tests"])))
    return record


def summarize(folder: Path) -> dict:
    rows = json.loads((folder / "promptfoo_results.json").read_text(encoding="utf-8"))["results"]["results"]

    def strategy(row):
        return (row.get("testCase") or {}).get("metadata", {}).get("strategyId") or "basic"
    return {"tests": len(rows), "errors": sum(r.get("failureReason") == 2 for r in rows),
            "attack_success": sum(r.get("failureReason") == 1 for r in rows),
            "by_strategy": {s: {"tests": n, "success": sum(1 for r in rows if strategy(r) == s and r.get("failureReason") == 1),
                                "errors": sum(1 for r in rows if strategy(r) == s and r.get("failureReason") == 2)}
                            for s, n in Counter(strategy(r) for r in rows).items()}}


def evaluate(model: str, ports) -> None:
    folder = ROOT / sc.model_dir(model)
    if (folder / "DONE").exists():
        print("skip (done)", model)
        return
    sc.archive_incomplete(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tests = load_yaml(GENERATE_DIR / "redteam.yaml")
    tests["targets"] = [{"id": f"ollama:chat:{model}", "label": "target"}]
    tests["description"] = f"study_v1 promptfoo - {model}"
    if sc.SMOKE_LIMIT:                       # smoke tests only: first N single-turn and first N multi-turn tests
        single = [t for t in tests["tests"] if strategy_of(t) not in MULTI_TURN][:sc.SMOKE_LIMIT]
        multi = [t for t in tests["tests"] if strategy_of(t) in MULTI_TURN][:sc.SMOKE_LIMIT]
        tests["tests"] = single + multi
    save_yaml(folder / "redteam.yaml", tests)
    started = sc.utc_now()
    wrapper = run_promptfoo(model, folder, ["redteam", "eval", "-c", str(folder / "redteam.yaml"), "--remote",
                                            "--no-cache", "--no-progress-bar", "-j", "4",
                                            "-o", str(folder / "promptfoo_results.json")], ports=ports)
    result = {"model": model, "started_utc": started, "finished_utc": sc.utc_now(),
              "promptfoo_exit": wrapper["exit_code"],      # 100 = some tests failed (= attacks succeeded); normal
              "duration_s": wrapper["duration_s"], **summarize(folder),
              **{k: wrapper[k] for k in ("target_requests", "target_errors", "rejected_model_events")}}
    result["ok"] = result["tests"] > 0 and result["target_errors"] == 0 and result["rejected_model_events"] == 0
    sc.write_json(folder / "result.json", result)
    sc.append_jsonl(ROOT / "progress.jsonl", result)
    if result["ok"]:
        (folder / "DONE").write_text(sc.utc_now(), encoding="utf-8")
    print(("OK  " if result["ok"] else "BAD ") + f"{model} tests={result['tests']} success={result['attack_success']} "
          f"errors={result['errors']}")


def summary() -> dict:
    out = {sc.model_dir(m): summarize(ROOT / sc.model_dir(m)) for m in sc.STUDY_MODELS
           if (ROOT / sc.model_dir(m) / "promptfoo_results.json").exists()}
    sc.write_json(ROOT / "summary.json", out)
    for model, s in out.items():
        print(f"{model:32s} {s['attack_success']} succeeded, {s['errors']} errors, {s['tests']} tests")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "select-multiturn", "eval", "summary"])
    parser.add_argument("model", nargs="?")
    parser.add_argument("--ports", default=None, help="target,judge,attacker ports (default 11500,11600,11601)")
    a = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if a.command == "generate":
        generate()
    elif a.command == "select-multiturn":
        native = GENERATE_DIR / "redteam_native_firstN.yaml"     # Promptfoo's original output is kept under this name
        if not native.exists():
            (GENERATE_DIR / "redteam.yaml").replace(native)
        select_multi_turn(native, GENERATE_DIR / "redteam.yaml", GENERATE_DIR / "multiturn_selection.json")
    elif a.command == "eval":
        assert a.model in sc.STUDY_MODELS, f"unknown model {a.model}"
        evaluate(a.model, sc.parse_ports(a.ports))
    else:
        summary()


if __name__ == "__main__":
    main()
