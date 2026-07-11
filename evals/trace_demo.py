"""Phase-4 observability: one LangSmith-traced run, read back and dumped, next
to askrepo's hand-rolled tracer.

    PROVIDER=openai secrun.sh .venv/bin/python evals/trace_demo.py

What askrepo's tracer records (askrepo/ops.py): a flat list of named spans with
millisecond durations — `{"retrieve": 120.4, "generate": 831.2}` — plus a few
attributes (tokens, cost), one JSON line to stderr. You get *that a step ran and
how long it took*. You do NOT get: the nested structure, the inputs/outputs at
each step, per-LLM-call token counts, or the exact prompt that was sent.

LangSmith traces every runnable in the LCEL chain automatically — no
instrumentation in our code — and this script reads the trace back through the
SDK so the difference is on the page, not just in a browser. It saves the tree
to evals/trace-sample.json and prints the run URL you can open/screenshot.

Region note (see PLAN gotchas): the account is APAC-sharded, so
LANGSMITH_ENDPOINT must point at apac.* or every call 403s. Set here explicitly.
"""

import json
import os
import sys
import time

# Tracing must be configured BEFORE the langchain imports below, so the tracer
# sees these at setup. The API key rides in from secrun; everything else we set
# explicitly so the run is reproducible.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://apac.api.smith.langchain.com")
os.environ.setdefault("LANGSMITH_PROJECT", "asklc-phase4")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tracers.context import collect_runs  # noqa: E402
from langchain_core.tracers.langchain import wait_for_all_tracers  # noqa: E402
from langsmith import Client  # noqa: E402

from asklc.rag import load_store, make_chain  # noqa: E402

QUESTION = "Which deep dive covers barge-in, and what is barge-in?"
OUT = os.path.join(os.path.dirname(__file__), "trace-sample.json")


def walk(run, depth=0):
    """Flatten the in-memory run tree into rows: name, type, ms, io sizes."""
    ms = None
    if run.end_time and run.start_time:
        ms = round((run.end_time - run.start_time).total_seconds() * 1000, 1)
    rows = [{
        "depth": depth,
        "name": run.name,
        "type": run.run_type,
        "ms": ms,
    }]
    for child in sorted(run.child_runs, key=lambda r: r.start_time or 0):
        rows.extend(walk(child, depth + 1))
    return rows


def main():
    if os.getenv("PROVIDER", "").lower() != "openai":
        raise SystemExit("Set PROVIDER=openai (this traces a real RAG run).")
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("No LANGSMITH_API_KEY — run under secrun.sh.")

    store, meta = load_store()
    chain = make_chain(store, meta["stack"], k=5)

    # collect_runs hands us the in-memory run tree (structure + latency), and
    # the run_id we can enrich from the server (tokens + cost).
    with collect_runs() as cb:
        answer = chain.invoke(QUESTION)
    wait_for_all_tracers()  # flush the background submission to LangSmith

    root = cb.traced_runs[0]
    rows = walk(root)
    print(f"\nanswer: {answer[:100]}...\n")
    print(f"=== LangSmith trace tree ({len(rows)} spans, auto-captured) ===")
    for r in rows:
        indent = "  " * r["depth"]
        ms = f"{r['ms']:>8.1f}ms" if r["ms"] is not None else "        —"
        print(f"  {ms}  {indent}{r['name']} [{r['type']}]")

    # Enrich the root from the server: total tokens + cost the local tree lacks.
    client = Client()
    enriched = {}
    for attempt in range(6):
        try:
            api_run = client.read_run(root.id)
            enriched = {
                "total_tokens": api_run.total_tokens,
                "prompt_tokens": api_run.prompt_tokens,
                "completion_tokens": api_run.completion_tokens,
                "total_cost": float(api_run.total_cost) if api_run.total_cost else None,
            }
            if api_run.total_tokens:
                break
        except Exception:
            pass
        time.sleep(2)  # replication lag before the enriched run is queryable

    try:
        url = client.get_run_url(run=root, project_name=os.environ["LANGSMITH_PROJECT"])
    except Exception:
        url = f"(open project {os.environ['LANGSMITH_PROJECT']!r}, run id {root.id})"

    print("\n=== server-side enrichment (what the hand-rolled tracer can't total) ===")
    print(f"  tokens: {enriched.get('total_tokens')} "
          f"({enriched.get('prompt_tokens')} in / {enriched.get('completion_tokens')} out)")
    print(f"  cost:   ${enriched.get('total_cost')}")
    print(f"  URL:    {url}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "question": QUESTION,
            "answer": answer,
            "run_id": str(root.id),
            "project": os.environ["LANGSMITH_PROJECT"],
            "url": url,
            "enrichment": enriched,
            "tree": rows,
        }, f, indent=1)
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
