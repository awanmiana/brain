"""Brain11 — one local edge AI system, three ~10.8M models (33M total).

Single package entry point. Everything else delegates to the existing
`brain11_*` modules so lattice/430M code is untouched.

    import brain11
    r = brain11.Brain11Router(device="cuda")
    print(r.code("write a function add(a, b) that returns a + b"))
"""

from __future__ import annotations

__version__ = "2.0.0"

# --- model architecture ---
from brain11_model import Brain, Brain11Config, load_model  # noqa: F401

# --- tokenizer ---
from brain11_tokenizer import (  # noqa: F401
    BOS_ID,
    EOS_ID,
    PAD_ID,
    TOKENIZER_PATH,
    UNK_ID,
    VOCAB_SIZE,
    decode,
    encode,
    get_tokenizer,
)

# --- router (one name: brain) ---
from brain11_router import (  # noqa: F401
    MODEL_FILES,
    Brain11Router,
    classify_task,
)

# --- reactive coder (the mechanism) ---
from brain11_reactive import CodeState, Hop, reactive_build  # noqa: F401

# --- planner ---
from brain11_planner import (  # noqa: F401
    Plan,
    dataflow_check,
    heal_code,
    insert_setup,
    plan_task,
)

# --- self-learning / healing loop ---
from brain11_loop import Cortex11, Doctor11, Healer, Memory11, Teacher11  # noqa: F401

TOTAL_PARAMS = 3 * Brain11Config().n_params()  # 32,379,264
PARAM_CAP = 33_000_000

__all__ = [
    "Brain", "Brain11Config", "load_model",
    "BOS_ID", "EOS_ID", "PAD_ID", "UNK_ID", "VOCAB_SIZE", "TOKENIZER_PATH",
    "encode", "decode", "get_tokenizer",
    "Brain11Router", "MODEL_FILES", "classify_task",
    "CodeState", "Hop", "reactive_build",
    "Plan", "plan_task", "dataflow_check", "insert_setup", "heal_code",
    "Cortex11", "Doctor11", "Healer", "Memory11", "Teacher11",
    "TOTAL_PARAMS", "PARAM_CAP", "__version__",
]


def model_dir():
    """Where model artifacts live. SEPARATE from source. Not in git."""
    import os
    from pathlib import Path
    env = os.environ.get("BRAIN11_MODEL_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent.parent
    return here / "models"
