"""Promptfoo on Llama 3 70B Instruct (Amazon Bedrock) (meta.llama3-70b-instruct-v1:0).

Runs the exact Promptfoo steps used in the study for this one target model, behind the logging proxies of
common/study_common.py (temperature 0, seed 42, context 4,096, max 1,024 output tokens; judge GPT-OSS-120B,
attacker Qwen3-Next-80B). Every step is resumable: finished parts are skipped on a re-run.

Run with any Python 3.10+ with PyYAML (Promptfoo itself runs on Node.js):
    python promptfoo_llama3_70b.py                      # main port set 11500 / 11600 / 11601
    python promptfoo_llama3_70b.py 11510,11610,11611    # second port set, to run in parallel with another scan

Output: runs/study_v1/promptfoo*/ below the repository root (see README.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import study_common as sc  # noqa: E402
import promptfoo_scan  # noqa: E402

MODEL = "meta.llama3-70b-instruct-v1:0"
PORTS = sc.parse_ports(sys.argv[1] if len(sys.argv) > 1 else None)


def select_multiturn_once() -> None:
    """Spread the 10 multi-turn tests over 10 seeded plugins instead of Promptfoo's first-N choice (D61)."""
    native = promptfoo_scan.GENERATE_DIR / "redteam_native_firstN.yaml"   # Promptfoo's original output kept here
    if not native.exists():
        (promptfoo_scan.GENERATE_DIR / "redteam.yaml").replace(native)
    promptfoo_scan.select_multi_turn(native, promptfoo_scan.GENERATE_DIR / "redteam.yaml",
                                     promptfoo_scan.GENERATE_DIR / "multiturn_selection.json")


def main() -> None:
    assert MODEL in sc.STUDY_MODELS
    promptfoo_scan.generate()                 # hosted generation, once; skipped when done
    select_multiturn_once()                   # seeded multi-turn spread over 10 plugins (D61)
    promptfoo_scan.evaluate(MODEL, PORTS)     # run the generated tests against this model
    promptfoo_scan.summary()


if __name__ == "__main__":
    main()
