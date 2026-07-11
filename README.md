# askrepo-langchain — the same app, built on the framework

The capstone ([askrepo](https://github.com/alexvervloet/ask-my-repo)) builds
repo-Q&A **from scratch**: hand-rolled chunker, its own index format, its own
agent loop. This project rebuilds the same core on **LangChain + LangGraph**
and answers the question job descriptions keep implying: *what does the
framework actually buy — and cost — versus code you control?*

The deliverable is not the app — it's **[COMPARISON.md](COMPARISON.md)**:

- a stage-by-stage mapping (askrepo module ↔ LangChain component, with LOC),
- the capstone's gold questions evaluated against **both** implementations,
- an honest list of what the abstraction bought and where it leaked.

Why this shape: "LangChain" appears in half the AI-engineering JDs, and
LangGraph is the part production teams actually run. Having built RAG from
scratch first, the port makes the framework legible instead of magical —
every LangChain component maps to code that already exists by hand in the
capstone. When the two implementations disagree (different chunks, different
retrieval, different eval scores), the *disagreement* is the finding.

## Quickstart (keyless)

```bash
/usr/local/bin/python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python check_setup.py
python -m asklc.smoke   # full LCEL chain + LangGraph graph on fake models —
                        # no API key, no network, no cost
```

The smoke run uses `FakeListChatModel` and `DeterministicFakeEmbedding` and
says so loudly — same rule as every other repo here: keyless never silently
pretends to be real.

**Real providers** reuse askrepo's env contract (`PROVIDER=claude|openai`).
Keys live in the macOS Keychain and are injected per-invocation:

```bash
secrun python -m asklc index ../DeepDives/deep-dive-capstone
secrun python -m asklc ask "where is retrieval implemented?"
```

## Layout

- `asklc/` — the port (grows phase by phase; see [PLAN.md](PLAN.md))
- `COMPARISON.md` — the write-up this project exists to produce
- [`LESSONS.md`](LESSONS.md) — engineering lessons from the port (framework
  defaults, coincidental eval parity, durable persistence, bought vs hand-rolled
  tracing)
- corpus + gold questions are **reused from the capstone**
  (`../DeepDives/deep-dive-capstone/fixtures/` and `evals/`) so both
  implementations are judged on identical inputs
