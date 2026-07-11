"""Non-secret config for the LangChain port, same env contract as askrepo.

askrepo/config.py reads PROVIDER/MODEL from .env with std-lib only so a fresh
clone runs before any `pip install`. We keep the *contract* identical — same
env var names, same `mock` default, real env wins over .env — so both
implementations are driven the same way and any behavioural difference is the
framework's, not the wiring's.

The one addition is CHROMA_DIR: askrepo saves a single JSON index; the port
persists a Chroma collection to a directory instead (see rag.py). That's a
difference in the *container*, not the idea — recorded in COMPARISON.md.
"""

import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "PROVIDER": "mock",   # mock|claude|openai — mock is keyless (fake models)
    "MODEL": "",          # each provider picks its own default; set to override
    "K": "5",             # chunks to retrieve — askrepo's explicit default (Chroma's is 4)
}


def _read_env_file(path):
    values = {}
    if not os.path.exists(path):
        return values
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def load_config():
    """Merge defaults <- .env <- real environment (strongest last)."""
    config = dict(DEFAULTS)
    config.update(_read_env_file(os.path.join(HERE, ".env")))
    for key in config:
        val = os.getenv(key)
        if val:
            config[key] = val
    config["PROVIDER"] = config["PROVIDER"].strip().lower()
    return config


# Where the persisted Chroma collection lives (gitignored). askrepo's analogue
# is index/index.json; ASKLC_CHROMA lets an alternate index sit beside it, the
# same way askrepo's ASKREPO_INDEX does, so eval runs can keep both on disk.
CHROMA_DIR = os.getenv("ASKLC_CHROMA") or os.path.join(HERE, ".chroma")
COLLECTION = "asklc"

# The capstone this project ports and compares against, and the corpus it
# indexes. Both default to a sibling checkout — the layout the README documents
# (../DeepDives/deep-dive-capstone next to this repo) — and are overridable via
# env for any other arrangement, so nothing hard-codes a personal path.
CAPSTONE_ROOT = os.getenv("ASKLC_CAPSTONE") or os.path.normpath(
    os.path.join(HERE, "..", "DeepDives", "deep-dive-capstone")
)
CORPUS_ROOT = os.getenv("ASKLC_CORPUS") or os.path.dirname(CAPSTONE_ROOT)
