# askrepo vs LangChain — the same app twice, honestly

**askrepo** is a command-line tool that answers questions about a code
repository with grounded, cited answers: ask *"where is retrieval
implemented?"* and it replies from the actual files, each claim tagged
`(path:line)`, or declines with *"Not in this corpus."* when the answer isn't
there. Its [capstone version](https://github.com/alexvervloet/ask-my-repo) is
built **from scratch** — a hand-written chunker, its own JSON index, a
vector+keyword hybrid retriever, and a hand-rolled agent loop, no framework.

This project (**asklc**) rebuilds the identical tool on **LangChain +
LangGraph** and measures the two against each other, to answer the question job
descriptions keep implying: *what does the framework actually buy — and cost —
versus code you control?* Everything below is evidence from running both, not
opinion. Depth lives in [PLAN.md](PLAN.md) (per-phase tables, running gotcha
log) and [LESSONS.md](LESSONS.md); this document is the standalone verdict.

## The setup

- **The app.** Index a corpus → ask → retrieve the most relevant chunks →
  answer with `(path:line)` citations, or decline when the corpus doesn't
  contain it. There's also an **agent** mode that answers by *searching*
  (grep/read tools) instead of embedding.
- **The corpus.** A 16-repo technical series (~400 markdown + python files).
  Both implementations index the byte-identical file set.
- **The gold set.** 40 questions in four categories — *locator* ("which part
  covers X?"), *concept*, *code* (about specific functions), *cross-dive* —
  each with expected files and key points; 8 are deliberately unanswerable and
  should be declined. Reused verbatim from the capstone.
- **The controls.** Same corpus, same models (`gpt-4o-mini` answers,
  `text-embedding-3-small` embeds), same `k=5`, and the **prompt contract
  copied verbatim** into the port. One shared `gpt-4o-mini` judge grades both
  arms. So any behavioral difference is the framework's, not the setup's.
- **What "from scratch" vs "framework" means, concretely:**

  | | from scratch (askrepo) | framework (asklc) |
  |---|---|---|
  | load | `os.walk` + `open` | `TextLoader` |
  | chunk | structure-aware, line-tracked | `RecursiveCharacterTextSplitter` |
  | store | one JSON file, vectors inline | `Chroma` |
  | retrieve | cosine **+ BM25**, blended | `as_retriever` (vector-only) |
  | answer | assemble + prompt + stream | LCEL `retrieve \| prompt \| model \| parse` |
  | agent | a hand-written `while` loop | a LangGraph `StateGraph` + checkpointer |

## Stage-by-stage mapping

Functional lines per stage (comments/docstrings excluded), and the one thing
that differs most. Full detail + the defaults hunt is in PLAN.md phase 1.

| stage | from scratch | framework | headline difference |
|---|---|---|---|
| load | ~30 | ~25 | `TextLoader` needs the **sunset** `langchain-community`; the walk/filter is still yours either way |
| chunk | ~55 | ~25 | **characters, not lines**; no `(path:line)` citations at all unless you opt into position tracking |
| embed | ~25 | ~6 | one call vs a batch loop — but you **can't price the build**, the token count isn't exposed |
| store | ~15 | ~13 | Chroma defaults to **L2, not cosine** — it changes rankings, silently |
| retrieve | ~90 | ~2 | the 90 lines bought a **hybrid**; the stock retriever is vector-only |
| answer | ~50 | ~30 | LCEL is tidy; token-budgeted assembly has no default and is DIY |

**The surprise is that the LOC win is lopsided and misleading.** Retrieval
collapses from ~90 lines to ~2 — but those 90 lines *were* the vector+BM25
hybrid, and the eval below shows it earns real wins. Meanwhile load and chunk
barely shrink, because the directory walk, the citation line-recovery (the
splitter tracks a character offset, not a line), and the cosine override all had
to be written back by hand. So "the framework writes less code" holds only at
the one stage where writing less code also means *doing less* — and that stage
is the one where doing less measurably costs you.

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
same corpus. askrepo runs its real hybrid retrieval (blend=0.7); asklc runs the
stock vector `as_retriever` — because vector-only *is* the framework default
we're measuring. (`evals/parity.py`, `evals/parity.run.json`, 2026-07-11.)

| metric | askrepo | asklc |
|---|---|---|
| judged correctness (32 answerable) | 0.814 | 0.800 |
| decline accuracy (8 negatives) | 1.000 | 1.000 |
| retrieval hit@k (an expected file made the top k) | 0.857 | 0.857 |
| tokens/answer p50 (in+out) | 2546 | 2419 |
| citation resolve / match | 0.881 / 0.667 | 0.951 / 0.732 |

**The parity is real but coincidental — and that's the finding.** hit@k is
identical to three decimals, but **12 of 40 questions behave differently**; the
differences cancel. Two mechanisms pull opposite ways:

- **askrepo's BM25 hybrid** wins when the answer sits in a specific file named
  by exact terms. On `code-07` ("when does the agent loop stop?") asklc's
  vector-only retriever returned the *same README chunk five times* and
  declined; askrepo's keyword signal surfaced the exact `EXERCISES.md` line.
- **asklc's larger character-window chunks** (from
  `RecursiveCharacterTextSplitter`) win when the answer is one line inside a
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

The honest answer isn't "framework vs no framework" — it's **which layer**.

**Reach for the framework (specifically LangGraph) for the agent/runtime.** The
strongest result in this whole comparison is durable persistence and
cross-process resume: a real capability the hand-rolled loop *cannot* have
without building a serializer, and one that came with per-node retries,
streaming, and checkpoint time-travel attached. If you're shipping anything
stateful — an agent that must survive a crash, a long tool loop, a resumable
job — LangGraph earns its weight. That's the part production teams actually run,
and it's where the abstraction is buying capability, not just syntax.

**Keep the RAG core thin and explicit — hand-rolled, or a minimal vector
library.** The RAG port did not buy accuracy; it *tied*. And it cost the hybrid
retrieval that measurably wins questions, the ability to price an index build,
and a stack of silent defaults (L2 not cosine, k=4, no citations) you have to
discover in source. When you care about specific retrieval behavior — hybrid
search, custom chunking, cost visibility — the framework's convenience is
working against you, hiding exactly the knobs you need. `LangChain`-the-RAG-chain
was the weakest value in this project; `LangGraph`-the-runtime was the strongest.

**And regardless of which you pick:** the framework's defaults are decisions
someone else made for you, often invisibly and often not the one you'd choose.
Read the source for the numbers, diff your evals per-question, and never let a
matched aggregate convince you two systems are equivalent — here it hid twelve
questions' worth of offsetting differences. The framework is a fine place to
start and a poor place to stop reading.
