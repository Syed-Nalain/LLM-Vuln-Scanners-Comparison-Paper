"""Augustus on Llama 3 70B Instruct (Amazon Bedrock) (meta.llama3-70b-instruct-v1:0).

Runs the exact Augustus steps used in the study for this one target model, behind the logging proxies of
common/study_common.py (temperature 0, seed 42, context 4,096, max 1,024 output tokens; judge GPT-OSS-120B,
attacker Qwen3-Next-80B). Every step is resumable: finished parts are skipped on a re-run.

Run with any Python 3.10+ with PyYAML (Augustus is a Go binary):
    python augustus_llama3_70b.py                      # main port set 11500 / 11600 / 11601
    python augustus_llama3_70b.py 11510,11610,11611    # second port set, to run in parallel with another scan

Output: runs/study_v1/augustus*/ below the repository root (see README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import study_common as sc  # noqa: E402
import augustus_scan  # noqa: E402

MODEL = "meta.llama3-70b-instruct-v1:0"
PORTS = sc.parse_ports(sys.argv[1] if len(sys.argv) > 1 else None)


def main() -> None:
    assert MODEL in sc.STUDY_MODELS
    augustus_scan.scan(MODEL, PORTS, set())  # 26 native probes + GOAT, Crescendo, TAP, PAIR
    augustus_scan.summarize_model(MODEL)


if __name__ == "__main__":
    main()
