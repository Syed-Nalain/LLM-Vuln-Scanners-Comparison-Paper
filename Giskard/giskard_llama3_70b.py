"""Giskard on Llama 3 70B Instruct (Amazon Bedrock) (meta.llama3-70b-instruct-v1:0).

Runs the exact Giskard steps used in the study for this one target model, behind the logging proxies of
common/study_common.py (temperature 0, seed 42, context 4,096, max 1,024 output tokens; judge GPT-OSS-120B,
attacker Qwen3-Next-80B). Every step is resumable: finished parts are skipped on a re-run.

Run with Giskard's Python (GISKARD_PYTHON):
    python giskard_llama3_70b.py                      # main port set 11500 / 11600 / 11601
    python giskard_llama3_70b.py 11510,11610,11611    # second port set, to run in parallel with another scan

Output: runs/study_v1/giskard*/ below the repository root (see README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import study_common as sc  # noqa: E402
import giskard_scan  # noqa: E402

MODEL = "meta.llama3-70b-instruct-v1:0"
PORTS = sc.parse_ports(sys.argv[1] if len(sys.argv) > 1 else None)


def main() -> None:
    assert MODEL in sc.STUDY_MODELS
    giskard_scan.generate()                   # scenario suites, once; existing suites are reused
    giskard_scan.scan(MODEL, PORTS)           # 98 single-turn + 10 multi-turn scenarios
    giskard_scan.summary()


if __name__ == "__main__":
    main()
