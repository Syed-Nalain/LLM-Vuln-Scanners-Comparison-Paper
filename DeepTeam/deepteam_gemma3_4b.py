"""DeepTeam on Gemma 3 4B (Ollama, local) (gemma3:4b).

Runs the exact DeepTeam steps used in the study for this one target model, behind the logging proxies of
common/study_common.py (temperature 0, seed 42, context 4,096, max 1,024 output tokens; judge GPT-OSS-120B,
attacker Qwen3-Next-80B). Every step is resumable: finished parts are skipped on a re-run.

Run with DeepTeam's Python (DEEPTEAM_PYTHON):
    python deepteam_gemma3_4b.py    # second port set 11510 / 11610 / 11611 (fixed, as in the study)

Output: runs/study_v1/deepteam*/ below the repository root (see README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import study_common as sc  # noqa: E402
import deepteam_scan  # noqa: E402

MODEL = "gemma3:4b"


def main() -> None:
    assert MODEL in sc.STUDY_MODELS
    deepteam_scan.simulate()                  # single-turn attacks, simulated once; skipped when done
    deepteam_scan.attack(MODEL)               # send the saved attacks + 10 multi-turn conversations
    deepteam_scan.summary()


if __name__ == "__main__":
    main()
