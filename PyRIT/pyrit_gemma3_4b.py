"""PyRIT on Gemma 3 4B (Ollama, local) (gemma3:4b).

Runs the exact PyRIT steps used in the study for this one target model, behind the logging proxies of
common/study_common.py (temperature 0, seed 42, context 4,096, max 1,024 output tokens; judge GPT-OSS-120B,
attacker Qwen3-Next-80B). Every step is resumable: finished parts are skipped on a re-run.

Run with PyRIT's Python (PYRIT_PYTHON):
    python pyrit_gemma3_4b.py                      # main port set 11500 / 11600 / 11601
    python pyrit_gemma3_4b.py 11510,11610,11611    # second port set, to run in parallel with another scan

Output: runs/study_v1/pyrit*/ below the repository root (see README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import study_common as sc  # noqa: E402
import pyrit_scan  # noqa: E402

MODEL = "gemma3:4b"
PORTS = sc.parse_ports(sys.argv[1] if len(sys.argv) > 1 else None)


def main() -> None:
    assert MODEL in sc.STUDY_MODELS
    pyrit_scan.scan(MODEL, PORTS)             # 148 single-turn attacks (C1, C3, C6) + 10 multi-turn
    pyrit_scan.summary()


if __name__ == "__main__":
    main()
