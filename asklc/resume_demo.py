"""Durable checkpoint/resume — the capability the hand-rolled loop can't have.

The claim: an agent mid-task can survive its process dying and continue in a
*fresh* process, because LangGraph persists the conversation to a checkpointer,
not to a Python variable. askrepo's `while` loop keeps `messages` in memory;
kill it and the work is gone. This demo proves the difference, keyless and
deterministically, in TWO separate processes sharing one SQLite file:

    python -m asklc.resume_demo start    # runs one tool call, checkpoints, EXITS
    python -m asklc.resume_demo resume    # fresh process, finishes from the disk

The model is a scripted fake (no key, no network): in `start` it asks to grep
the real corpus (a real tool call through askrepo's real harness, so a real
observation gets checkpointed); in `resume` it emits the final answer. The
point isn't the model's intelligence — it's that process 2 has a brand-new
model and empty memory, and the only thing carrying the half-finished task
across the gap is the checkpoint on disk.
"""

import os
import sqlite3
import sys

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from asklc.agent import build_agent, initial_state
from asklc.config import HERE

CORPUS = "/Users/alex/Documents/WebDev/AI/DeepDives"
DB_PATH = os.path.join(HERE, ".agent-resume.sqlite")
THREAD = "resume-demo"
QUESTION = "Which deep dive covers barge-in?"


class ScriptedModel:
    """Stands in for a real chat model: returns a pre-set AIMessage, ignores
    input. `bind_tools` is a no-op (a fake doesn't need the schema). Each
    process constructs its own — mirroring that the model has no memory; only
    the checkpointer does."""

    def __init__(self, message):
        self._message = message

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._message


def _saver():
    # check_same_thread=False: the saver may touch the conn from a worker thread
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)


def _harness():
    from asklc.agent import _askrepo  # ensures the capstone is on sys.path

    _, H = _askrepo()  # reuse the real read-only boundary
    return H.default_harness(CORPUS)


def start():
    # The model's first move: one real grep against the corpus.
    model = ScriptedModel(
        AIMessage(
            content="",
            tool_calls=[{"name": "grep", "args": {"pattern": "barge-in"}, "id": "c1"}],
        )
    )
    touched = set()
    # interrupt_after=["act"]: stop the instant the observation is checkpointed.
    app = build_agent(model, _harness(), touched, _saver(), interrupt_after=["act"])
    config = {"configurable": {"thread_id": THREAD}, "recursion_limit": 50}
    app.invoke(initial_state(QUESTION), config)

    state = app.get_state(config)
    print(f"[start] pid {os.getpid()} — ran {state.values['n_calls']} tool call(s), "
          f"grepped files: {sorted(touched)}")
    print(f"[start] checkpoint has {len(state.values['messages'])} messages; "
          f"next node: {state.next}")
    print(f"[start] persisted to {DB_PATH} — now EXITING mid-task.")


def resume():
    if not os.path.exists(DB_PATH):
        raise SystemExit("No checkpoint. Run `python -m asklc.resume_demo start` first.")
    # A brand-new model with the final answer — process 2 knows nothing except
    # what it reads back from the checkpoint.
    model = ScriptedModel(
        AIMessage(content="The Realtime Voice dive covers barge-in "
                          "(realtime-voice-deep-dive/README.md:1).")
    )
    app = build_agent(model, _harness(), set(), _saver())
    config = {"configurable": {"thread_id": THREAD}, "recursion_limit": 50}

    before = app.get_state(config)
    if not before.values:
        raise SystemExit("Checkpoint empty — did `start` run in this same repo?")
    print(f"[resume] pid {os.getpid()} — recovered {len(before.values['messages'])} "
          f"messages from disk, {before.values['n_calls']} tool call(s) already done")
    print(f"[resume] paused before node: {before.next} — continuing…")

    final = app.invoke(None, config)  # None = resume from the checkpoint
    print(f"[resume] FINAL ANSWER: {final['messages'][-1].content}")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    if phase == "start":
        # fresh run each demo: clear any stale checkpoint
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        start()
    elif phase == "resume":
        resume()
    else:
        raise SystemExit("usage: python -m asklc.resume_demo start|resume")


if __name__ == "__main__":
    main()
