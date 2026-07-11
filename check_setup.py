#!/usr/bin/env python3
"""Verify the environment for askrepo-langchain. Makes no paid call — the
phase 0 smoke path runs entirely on fake models."""

import sys
from pathlib import Path

OK, BAD = "  \033[32m✓\033[0m", "  \033[31m✗\033[0m"
failures = 0


def check(label: str, ok: bool, fix: str = "") -> None:
    global failures
    print(f"{OK if ok else BAD} {label}" + ("" if ok else f"\n      fix: {fix}"))
    failures += 0 if ok else 1


print("askrepo-langchain setup check\n")

check(
    f"Python {sys.version_info.major}.{sys.version_info.minor} (need 3.11+)",
    sys.version_info >= (3, 11),
    "install a newer Python",
)

for pkg in (
    "langchain",
    "langgraph",
    "langchain_chroma",
    "langchain_community",  # TextLoader for the phase-1 `load` stage (sunset; see PLAN)
    "langchain_anthropic",
    "langchain_voyageai",
    "langchain_openai",
    "dotenv",
):
    try:
        __import__(pkg)
        check(f"package: {pkg}", True)
    except ImportError:
        check(f"package: {pkg}", False, "pip install -r requirements.txt")

# Phases 1–2 reuse the capstone's corpus and gold questions verbatim.
capstone = Path(__file__).parent.parent / "DeepDives" / "deep-dive-capstone"
check(
    f"capstone corpus: {capstone / 'fixtures'}",
    (capstone / "fixtures").is_dir(),
    "clone/locate deep-dive-capstone next to this repo's parent",
)
check(
    f"capstone gold questions: {capstone / 'evals'}",
    (capstone / "evals").is_dir(),
    "clone/locate deep-dive-capstone next to this repo's parent",
)

print(
    "\nNo API-key check here on purpose: keys live in the Keychain and real\n"
    "provider runs go through `secrun` (phase 1+). Keyless smoke:\n"
    "  python -m asklc.smoke"
)

sys.exit(1 if failures else 0)
