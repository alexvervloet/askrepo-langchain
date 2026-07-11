# askrepo vs LangChain — the same app twice, honestly

*(Filled in as the phases in [PLAN.md](PLAN.md) complete. Skeleton below is
the contract for what this document must answer.)*

## The setup

Same corpus, same gold questions, same provider and k. One implementation is
the capstone's from-scratch code; the other is LangChain + LangGraph. Any
difference in behavior is attributable to the framework.

## Stage-by-stage mapping

*(final table from PLAN.md phase 1, plus prose on the surprises)*

## What the framework bought

- **Durable agent state — the one thing the hand-rolled loop structurally
  can't have.** askrepo's agent keeps its conversation in a local `messages`
  variable; kill the process and the task is gone. The LangGraph port checkpoints
  after every super-step, and with `SqliteSaver` that survives the process. Proven
  in two separate processes (`asklc/resume_demo.py`): process 1 runs a real tool
  call, checkpoints, and exits mid-task; process 2 — different pid, empty memory,
  fresh model — recovers the state from disk and finishes. This is not a LOC
  saving; it is a capability that wasn't reachable before without hand-building a
  serializer.
- **Per-node retry / timeout / cache as one-liner kwargs.** `add_node(...,
  retry_policy=RetryPolicy(max_attempts=n))` (verified in `langgraph.types`),
  where askrepo hand-wrote `with_retry` and applied it to embeddings only.
- **Components swapped in one line.** The whole RAG pipeline is an LCEL
  expression (`retrieve | prompt | model | parse`); swapping Chroma, the
  embedder, or the chat model is a one-object change, vs askrepo's hand-written
  index format + provider classes.
- **Free plumbing:** query `input_type` auto-set for Voyage, `graph.stream()`
  node-level progress, `get_state_history` checkpoint time-travel — all things
  askrepo wrote by hand or doesn't have. (Full capability table in PLAN.md
  phase 3.)
- **A bought trace vs a hand-rolled one.** With `LANGSMITH_TRACING=true` and no
  instrumentation of ours, LangSmith captured all 9 runnables of the RAG chain
  nested, with per-step inputs/outputs, tokens and auto-priced cost, persisted
  and shareable — and it revealed the retriever (1819 ms) outweighs the LLM
  (1318 ms), a split askrepo's two hand-wrapped spans hide. askrepo's tracer
  still wins on zero-dependency and no-data-leaves-the-box; LangSmith wins on
  structure, I/O capture, and persistence. (Trace + visual: PLAN.md phase 4.)

## What the framework cost

- **Silent defaults that were wrong for the task, found only in source.** Chroma
  ranks by **L2, not cosine** (buried in chromadb, not surfaced by
  `langchain-chroma`) — it changes rankings, not just numbers; the retriever
  defaults to **k=4** not 5; `RecursiveCharacterTextSplitter` tracks **no
  position** unless you opt in, and then it's a char offset, not a line. Every
  one came from reading installed source, not docs (PLAN.md phase 1).
- **No hybrid retrieval in the box.** askrepo's BM25+vector blend has no default
  analogue; the stock retriever is vector-only and, on `code-07`, returned the
  *same README five times* and declined a question askrepo answered. Matching it
  means wiring `EnsembleRetriever` + `BM25Retriever` yourself.
- **Version churn during the project itself.** The canonical `TextLoader` lives
  in `langchain-community`, which prints a "being sunset / no longer maintained"
  warning on import. A *durable* checkpointer is a separate package
  (`langgraph-checkpoint-sqlite`) from the `langgraph` you install. You feel the
  ecosystem moving while you build on it.
- **Lost visibility.** `Embeddings.embed_documents` returns only vectors — the
  per-build token/cost line askrepo prints can't be reproduced without bypassing
  the abstraction and re-tokenizing.

## Eval results

40 gold questions, both implementations, one shared gpt-4o-mini judge, k=5,
same DeepDives corpus. askrepo runs its real hybrid retrieval (blend=0.7);
asklc runs the stock vector `as_retriever` — because vector-only *is* the
framework default we're measuring. (`evals/parity.py`, `evals/parity.run.json`,
2026-07-11.)

| metric | askrepo | asklc |
|---|---|---|
| judged correctness (32 answerable) | 0.814 | 0.800 |
| decline accuracy (8 negatives) | 1.000 | 1.000 |
| retrieval hit@k | 0.857 | 0.857 |
| tokens/answer p50 (in+out) | 2546 | 2419 |
| citation resolve / match | 0.881 / 0.667 | 0.951 / 0.732 |

**The parity is real but coincidental — and that's the finding.** hit@k is
identical to three decimals, but **12 of 40 questions behave differently**; the
differences cancel. Two mechanisms pull opposite ways:

- **askrepo's BM25 hybrid** wins when the answer sits in a specific file named
  by exact terms. On `code-07` ("when does the agent loop stop?") asklc's
  vector-only retriever returned the *same README chunk five times* and
  declined; askrepo's keyword signal surfaced the exact `EXERCISES.md` line.
- **asklc's larger RCTS chunks** win when the answer is one line inside a
  function that askrepo's per-`def` chunker splits off. On `code-03`
  (cosine_similarity of an all-zero vector) asklc's 1500-char chunk kept the
  whole function together and cited it; askrepo retrieved a different slice of
  the same file and declined.

So a stock vector RAG *matched* a hand-tuned hybrid on this corpus — by trading
wins question-for-question, not because the pipelines are equivalent. asklc even
edges citation quality (0.951 vs 0.881 resolve): bigger chunks carry a wider,
correctly-numbered line span, so a cited line lands inside a real chunk more
often. The 0.814-vs-0.800 correctness gap is within judge nondeterminism; the
signal is the churn and its two named causes.

**The practitioner takeaway:** the framework's default retriever has two traps
this eval surfaced — it returns near-duplicate chunks (no diversity; fix with
`search_type="mmr"`) and it has no keyword channel for exact-match terms (fix
with `EnsembleRetriever` + `BM25Retriever`). Neither is on by default, and the
aggregate score won't warn you — only the per-question view does.

## When I'd choose which

*(the paragraph an interviewer is actually asking for)*
