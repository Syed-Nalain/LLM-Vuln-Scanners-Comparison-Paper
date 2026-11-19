"""Garak on Qwen2.5 7B Instruct (Ollama, local) (qwen2.5:7b).

Runs the exact Garak steps used in the study for this one target model, behind the logging proxies of
common/study_common.py (temperature 0, seed 42, context 4,096, max 1,024 output tokens; judge GPT-OSS-120B,
attacker Qwen3-Next-80B). Every step is resumable: finished parts are skipped on a re-run.

Run with Garak's Python (GARAK_PYTHON):
    python garak_qwen2_5_7b.py    # main port set 11500 / 11600 / 11601 (fixed, as in the study)

Output: runs/study_v1/garak*/ below the repository root (see README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import study_common as sc  # noqa: E402
import garak_scan  # noqa: E402

MODEL = "qwen2.5:7b"


def main() -> None:
    assert MODEL in sc.STUDY_MODELS
    garak_scan.run_static([MODEL], set())     # 24 static probes, forced caps, primary detector
    garak_scan.run_goat([MODEL])              # GOAT, 10 JailbreakBench goals, max 5 calls each
    garak_scan.static_summary()               # recompute results from the saved reports


if __name__ == "__main__":
    main()
