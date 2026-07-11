# Plan — phases and definitions of done

Every phase ends with a filled table or a passing run, not a feeling. The
running gotcha log at the bottom feeds COMPARISON.md — write things down the
moment they surprise you.

## Phase 0 — plumbing (done when the smoke run passes keyless)

- [x] `python check_setup.py` all green
- [x] `python -m asklc.smoke` — an LCEL retrieval chain **and** a LangGraph
      graph with a checkpointer, on fake models: no key, no network
- [x] record the installed `lang*` versions below — LangChain's release churn
      is part of what this project measures, so pin the snapshot in writing

| package | version installed (2026-07-11) |
|---|---|
| langchain | 1.3.13 |
| langchain-core | 1.4.9 |
| langgraph | 1.2.9 |
| langgraph-checkpoint | 4.1.1 |
| langchain-chroma | 1.1.0 |
| langchain-anthropic | 1.4.8 |
| langchain-voyageai | 0.4.0 |
| langchain-openai | 1.3.5 |

## Phase 1 — RAG core port (done when the mapping table is full)

Load the capstone's `fixtures/` corpus with LangChain loaders →
`RecursiveCharacterTextSplitter` → same embeddings as askrepo (Voyage on the
claude stack) → Chroma → LCEL retrieval chain. Same corpus, same provider,
same k — differences must come from the framework, not the setup.

| stage | askrepo (module, ~LOC) | LangChain (component, ~LOC) | defaults that differ |
|---|---|---|---|
| load | `indexer.collect_chunks` ~30 (os.walk, ext allow-list, skip-dirs, open) | `corpus.load_documents` ~25 (`TextLoader` per file, *same* walk) | `TextLoader` lives in `langchain-community`, which prints a "being sunset / no longer maintained" warning on import; `DirectoryLoader`'s glob+exclude can't cleanly express the allow-list, so the walk stays hand-rolled either way |
| chunk | `indexer.chunk_markdown` / `chunk_python` + `_windows`/`_emit` ~55 (structure-aware: heading / `def`·`class`, line-tracked) | `corpus.chunk_documents` ~25 (`RecursiveCharacterTextSplitter`) | **characters not lines**; generic separator ladder `["\n\n","\n"," ",""]` vs heading / object boundaries; default size/overlap **4000/200 chars** (we set 1500/200) vs askrepo's explicit **60/10 lines**; `add_start_index` **defaults False** → no position tracking at all; when on it's a char *offset*, not a line → we convert offset→line by hand for citations |
| embed | `indexer.build_index` loop + `providers.embed` ~25 (batch 100, round 6dp, `input_type` by hand) | `rag.build_index` → `get_embeddings` ~6 (one call) | Voyage batch default **1000** vs 100; `truncation` **defaults True** vs askrepo never truncates; `input_type` auto-set (`document`/`query`) vs explicit; **no token count surfaced** through `embed_documents` → an index build can't be priced without reaching past the abstraction |
| store/index | `indexer.build_index` `json.dump` ~15 (one human-readable JSON, vectors inline) | `Chroma.from_documents` + `_write_meta` ~13 (persisted Chroma dir + sidecar) | container is a SQLite+HNSW dir vs one JSON file; the embedding **stack has no slot in Chroma** → carried in a sidecar so query-time uses the same model; **distance metric default L2** (chromadb HNSW) vs askrepo's cosine → overridden via `hnsw:space=cosine` |
| retrieve | `retrieve.py` cosine + BM25, min-max blended ~90 (hybrid, `blend=0.7`, `k=5`) | `store.as_retriever(search_kwargs={"k":k})` + `.invoke` ~2 | **vector-only** — no BM25 blend in the box (would need `EnsembleRetriever`+`BM25Retriever` to match), so exact-match terms lose the keyword signal; `k` **defaults 4** (`langchain_chroma.DEFAULT_K`) vs 5 → overridden; scores hidden unless you call `similarity_search_with_score`; no score threshold either way |
| assemble+answer | `answer.prepare` + `assemble.py` + `prompts.build_messages` + `providers.complete` ~50 | `rag.make_chain`/`answer` + `prompt.PROMPT` ~30 (LCEL `retrieve \| prompt \| model \| parse`) | token-budgeted greedy assembly (`assemble.py`) has **no framework default** — the k docs are just joined; streaming + `usage`/cost that `providers.complete` surfaces are extra wiring you add back in LCEL |

### Answers (the point of the phase)

**Which knobs did askrepo make explicit that the framework defaults silently?**
Almost all of them — and the silent default is often *not* what askrepo chose:

- **chunk size / overlap** — framework 4000/200 *chars*; askrepo 60/10 *lines*.
  At the default 4000, every file in a small corpus is one unsplit chunk.
- **separators** — framework `["\n\n","\n"," ",""]` (generic); askrepo splits on
  markdown headings and python `def`/`class`, so a chunk is one *thing*.
- **position tracking** — framework `add_start_index=False`, i.e. citations are
  impossible until you opt in, and even then it's a char offset, not a line.
- **distance metric** — Chroma defaults to **L2**; askrepo uses cosine. Silent,
  and it changes the ranking, not just the numbers. The nastiest default here.
- **k** — Chroma retriever defaults to **4**; askrepo's explicit default is 5.
- **hybrid blend** — askrepo's `blend=0.7` (vector+BM25) has *no* framework
  analogue; the stock retriever is vector-only, so this knob simply vanishes.
- **embedding batch / truncation / input_type** — Voyage defaults 1000 / True /
  auto; askrepo sets 100 / never / explicit.

**Where did I have to read LangChain *source* rather than docs?** Every default
in the table above came from source, not the component's docstring:

- `chunk_size=4000`, `chunk_overlap=200`, `separators=[…]` —
  `langchain_text_splitters/base.py` and `character.py`.
- `DEFAULT_K = 4` — `langchain_chroma/vectorstores.py`.
- Chroma's L2 distance default (`hnsw:space`) — down in **chromadb**, not
  surfaced by `langchain-chroma` at all; you learn it by getting wrong rankings.
- Voyage `batch_size=1000`, `truncation=True`, and the auto
  `document`/`query` `input_type` mapping — `langchain_voyageai/embeddings.py`;
  none of it is visible from the retriever API you actually call.
- `search_type="similarity"` default + allowed types —
  `langchain_core/vectorstores/base.py`.

## Phase 2 — eval parity (done when both columns are filled)

Run the capstone's gold questions against both implementations. Measured the
fair way: both answer with gpt-4o-mini at k=5, over the same DeepDives corpus,
with askrepo keeping its real hybrid retrieval (blend=0.7) and asklc using the
stock vector retriever — because "the framework default is vector-only" *is*
the difference. One shared gpt-4o-mini judge grades both (evals/parity.py,
run 2026-07-11, evals/parity.run.json).

| metric | askrepo | asklc (LangChain) |
|---|---|---|
| gold questions passed (mean judge score, 32 answerable) | **0.814** | **0.800** |
| decline accuracy (8 negative questions) | 1.000 | 1.000 |
| avg retrieval hit (gold chunk in context) | **0.857** | **0.857** |
| tokens per answer, p50 (in+out) | 2546 | 2419 |
| citation resolve / match | 0.881 / 0.667 | 0.951 / 0.732 |

Per category (askrepo / asklc): locator 0.96/0.96, concept 1.00/0.90,
code 0.56/**0.63**, cross-dive 0.50/0.50.

**Diagnosis — the aggregate parity is real but coincidental.** hit@k is
identical to three decimals, yet **12 of 40 questions behave differently** on
hit or correctness; the differences offset. Two mechanisms pull in opposite
directions, both traced to concrete questions:

1. *askrepo's BM25 hybrid wins* when the answer lives in a specific file named
   by exact terms that vector search buries. `code-07` ("when does the agent
   loop stop?"): asklc's vector-only retriever returned the *same README five
   times* (no diversity) and declined; askrepo's keyword blend surfaced
   `agents-deep-dive/EXERCISES.md:65` and answered. Same shape on `con-10`.
2. *asklc's larger RCTS chunks win* when the answer is one line inside a
   function that askrepo's per-`def` chunking splits away from the retrieved
   chunk. `code-03` (cosine_similarity on an all-zero vector): asklc's
   1500-char chunk kept the whole function — guard clause included — and cited
   `rag-deep-dive/rag/store.py:34`; askrepo retrieved a *different* slice of the
   same file, missed the `if norm_a == 0` line, and declined.

So the framework's stock vector RAG *matched* a hand-tuned hybrid here — not
because the pipelines are equivalent, but because coarser chunking and weaker
retrieval traded wins question-for-question. asklc even edges citation quality
(0.951 vs 0.881 resolve): bigger chunks carry a wider, correctly-numbered line
span, so a cited line lands inside a real chunk more often; askrepo's tight
chunks + windowing occasionally cite a line just past the retrieved span.
Correctness (0.814 vs 0.800) is inside the judge's own nondeterminism — the
*signal* is the per-question churn and its two named causes, not the tie.

## Phase 3 — agent on LangGraph (done when parity + resume both work)

Port the v05-agent tool loop to a `StateGraph`: explicit nodes for
plan/act/observe, tools bound to the model, checkpointer on. The tools and the
read-only boundary are askrepo's, reused verbatim (asklc/agent.py imports
`run_tool` + `default_harness`), so the *only* variable is who runs the loop.

- [x] **the agent completes the same tasks** — real gpt-4o-mini run answers the
      gold questions through plan→act→observe with citations (e.g. "barge-in" in
      3 tool calls; the all-zero cosine_similarity question — which RAG made
      askrepo *decline* in phase 2 — the agent answers by grepping store.py).
- [x] **kill mid-run, resume from the checkpoint** — `asklc/resume_demo.py`,
      keyless and deterministic, in two separate processes sharing one SQLite
      file: process 1 runs a real grep, checkpoints 4 messages, exits paused
      before `reason`; process 2 (different pid, empty memory, fresh model)
      recovers the 4 messages from disk and finishes. The hand-rolled loop keeps
      `messages` in a local variable — kill it and the task is gone.
- [x] **capability table** (below).

### Capability table — hand-rolled `while` vs LangGraph `StateGraph`

| capability | askrepo (hand-rolled) | LangGraph | what it cost to get |
|---|---|---|---|
| loop control | explicit `while n_calls < MAX` + a final forced-answer turn (~12 lines) | conditional edges + a `recursion_limit` safety net; the tool budget is still hand-tracked in state | ~a wash on LOC. LangGraph throws a `GraphRecursionError` for free if you forget a budget; you still write the budget itself |
| retries | none on the agent step — a transient API error kills the answer (askrepo has `with_retry` for embeddings only) | `add_node(..., retry_policy=RetryPolicy(max_attempts=n))`, plus `cache_policy` and `timeout`, as one-liner kwargs (verified in `langgraph.types`) | LangGraph: one kwarg per node. Hand-rolled: wrap each call yourself |
| persistence / resume | **none** — state is a Python variable; process death = lost task | checkpoint after every super-step; `SqliteSaver` = durable cross-process resume, **proven in two processes** | the headline win. Cost: one extra dep (`langgraph-checkpoint-sqlite`) + choosing a `thread_id`. The `while` loop can't get this without hand-building a serializer |
| streaming | `provider.complete` streams tokens; the agent path uses non-streamed `step` turns | `graph.stream()` streams state per node for free; token-level needs `astream_events` | comparable; LangGraph gives node-level progress for nothing |
| observability / replay | `AuditLog` + structured trace, hand-written | `get_state` / `get_state_history` replay every checkpoint (time-travel); LangSmith traces each node | LangGraph gives checkpoint time-travel for free — phase-4 territory |

**Net:** LangGraph bought durable persistence/resume (the capability the
hand-rolled loop *structurally* cannot have), plus per-node retry/timeout/cache
and state-level streaming as one-liners. It cost a heavier mental model
(StateGraph, reducers, super-steps), a separate package for a *durable*
checkpointer, and you still write the loop-budget logic yourself. Tool execution
and the security boundary were a straight reuse — identical in both.

## Phase 4 — observability (stretch)

- [x] One LangSmith-traced RAG run, read back through the SDK and dumped
      (`evals/trace_demo.py` → `evals/trace-sample.json`), next to askrepo's
      hand-rolled `Trace.span()` (askrepo/ops.py). Visual (the "screenshot"):
      artifact <https://claude.ai/code/artifact/c50563d4-9ea6-4490-bff5-33ab4054cca4>,
      plus the live run URL in the trace dump. (This environment has no browser
      to screenshot the LangSmith UI, so the artifact renders the same
      SDK-read data; the run URL is there to open/screenshot directly.)

**What the bought trace shows that the hand-rolled one doesn't (3 sentences):**
LangSmith traced all **nine** runnables in the LCEL chain automatically, with
none of our own tracing code, where askrepo records only the **two** spans we
remembered to wrap by hand. That granularity surfaces what a coarse span hides:
the vector **retriever (1819 ms)** — mostly the OpenAI query-embedding
round-trip — is **slower than the LLM call (1318 ms)**, a split askrepo's single
`retrieve` span can't show. And every step's real inputs and outputs (the
retrieved chunks, the exact assembled prompt) are captured and kept, queryable
and shareable by URL, where askrepo's trace is one JSON line to stderr that
scrolls away.

Honest scope: askrepo *does* record total tokens/cost — as flat trace
attributes — so that isn't the difference. The difference is the automatic
**nested structure**, **per-step I/O**, and **persistence/shareability**; the
hand-rolled tracer gives you timing on the spans you thought to wrap, and only
for as long as the terminal buffer lasts.

## Phase 5 — the write-up (done when COMPARISON.md stands alone)

- [x] COMPARISON.md complete: **bought / cost / when to choose which** — the
      "when to choose which" verdict is *which layer*, not which framework:
      LangGraph for the stateful agent/runtime (durable resume is a real
      capability), a thin/explicit RAG core for retrieval (the port tied on
      accuracy and cost the hybrid + cost-visibility + control over defaults).
- [x] cross-link — this repo's README headlines COMPARISON.md; a pointer added
      to the capstone's LEARNINGS (`ask-my-repo/LEARNINGS.md`). There is no
      top-level portfolio README in `AI/` to link from.
- [x] sanity check: rewrote the intro + "The setup" so a reader who never saw
      the capstone gets the app, corpus, gold set, and what "from scratch vs
      framework" means before any result; glossed hit@k and spelled out
      RecursiveCharacterTextSplitter. It stands alone.

## Notes / gotchas discovered along the way

(keep a running log here — this becomes the best part of COMPARISON.md)

- **LangSmith is region-sharded and nothing defaults to APAC.** The account
  lives in the APAC region, so the key only works against
  `https://apac.api.smith.langchain.com` — the SDK, the docs' curl examples,
  and `LANGSMITH_ENDPOINT`-less clients all default to US and return a bare
  403 "Forbidden" (not 401), which reads like a permissions bug, not a routing
  bug. A bogus key gets the *same* 403, so you can't binary-search it from
  status codes alone. Every LangSmith-touching run needs
  `LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com` exported.
  Verified 2026-07-11: traced a fake-model chain into project `asklc-smoke`
  and read the run back via `list_runs`. Key is in the Keychain as
  `deepdives:LANGSMITH_API_KEY` (secrun.sh injects it as an optional key).

- **Reading a trace back in-process: `collect_runs` + `wait_for_all_tracers`.**
  Tracing submits in the background, so a naive `list_runs` right after
  `invoke` races the upload and returns nothing. `collect_runs()` (from
  `langchain_core.tracers.context`) hands you the in-memory run tree
  immediately — structure + per-span latency, no network — and
  `wait_for_all_tracers()` blocks until the upload flushes so the server-side
  enriched run (token totals, auto-priced cost) is queryable via
  `Client.read_run`. The enriched fields lag a second or two behind the
  in-memory tree; retry `read_run` a few times. (phase 4)

- **Chroma's default distance is L2, not cosine — and nothing tells you.**
  `langchain-chroma` never mentions it; the default lives in chromadb's HNSW
  config (`hnsw:space="l2"`). askrepo ranks by cosine, so an unconfigured
  Chroma silently ranks in a *different geometry* — same vectors, different
  neighbours. Fixed by passing `collection_metadata={"hnsw:space":"cosine"}`
  to `Chroma.from_documents`. This is the one default that changes answers
  rather than just numbers. (phase 1)

- **`RecursiveCharacterTextSplitter` doesn't track position by default.**
  `add_start_index` defaults to `False`, so out of the box a chunk has no idea
  where it came from — citations are impossible until you opt in. And when you
  do, it's a *character offset* (`metadata["start_index"]`), not a line, so
  `(path:line)` citations need an offset→line conversion against the parent
  document that askrepo's line-tracking chunker never needed. (phase 1)

- **The stock vector retriever is vector-only.** `store.as_retriever()` has no
  BM25/keyword blend — askrepo's `blend=0.7` hybrid (great for module names,
  flags, error strings) has no framework analogue short of wiring up
  `EnsembleRetriever` + `BM25Retriever`. Default `k` is 4, not askrepo's 5.
  Expect this to be the main source of any phase-2 retrieval-score gap. (phase 1)

- **`langchain-community` is being sunset.** The canonical `TextLoader` for the
  `load` stage imports from `langchain-community`, which prints a
  DeprecationWarning on import: "being sunset / no longer actively maintained,
  migrate to standalone integration packages." Reading a text file the
  framework way pulls in a package its own maintainers are winding down (and
  `langchain-classic` + sqlalchemy as transitive deps). (phase 1)

- **Embeddings don't surface a token count.** `providers.embed` in askrepo
  returns `(vectors, tokens)` so every index build is priced. LangChain's
  `Embeddings.embed_documents` returns only vectors — the token/cost line
  askrepo prints on `index` can't be reproduced without bypassing the
  abstraction and re-tokenizing. (phase 1)

- **Phase-1 scope note.** The keyless (`PROVIDER=mock`) path is verified
  end-to-end: `index` → deterministic-fake embed → Chroma → `ask` → LCEL
  chain. A real Voyage/Claude run is deferred to phase 2, where eval parity
  actually needs it; the mapping/defaults analysis above (this phase's DoD)
  came from reading installed source, not from a paid run.

- **Vector-only retrieval fails silently by returning duplicates.** On
  `code-07` the stock `as_retriever(k=5)` returned the *same README chunk five
  times* — five near-identical vectors, no diversity — and the answer declined
  because the actual source file never made the top 5. askrepo's BM25 blend
  breaks that tie with a keyword signal; the LangChain equivalent is
  `search_type="mmr"` (maximal-marginal-relevance) or a post-retrieval dedupe,
  neither of which is the default. If you take the default retriever, budget
  for this. (phase 2)

- **Coarser chunks can beat finer ones — RCTS ~tied askrepo's structural
  chunker.** Intuition said askrepo's one-object-per-chunk splitting would win;
  in practice RCTS's 1500-char chunks sometimes won by keeping a whole function
  (guard clauses included) in a single retrievable unit, where the structural
  chunker split the answer line into a neighbouring chunk that didn't get
  retrieved. Structure-awareness is not free accuracy. (phase 2)

- **Eval parity was measured in ONE process, not against the frozen baseline.**
  The capstone's `baseline.run.json` predates this comparison and used a
  possibly different judge/corpus SHA. `evals/parity.py` re-runs *both*
  pipelines back-to-back with a single shared gpt-4o-mini judge, so the two
  columns can't drift on grader or corpus — the only variable is the pipeline.
  Cost of one full run: ~$0.05 (80 answers + ~64 judge calls). (phase 2)

- **A durable checkpointer is a SEPARATE package.** `langgraph` ships
  `MemorySaver` (in `langgraph-checkpoint`), which is in-memory — kill the
  process and the "checkpoint" is gone, so it can't demonstrate the one thing
  that matters vs the hand-rolled loop. Cross-process resume needs
  `langgraph-checkpoint-sqlite` (`SqliteSaver`), a separate install. The
  headline capability is real but it is not in the box you `pip install
  langgraph` for. (phase 3)

- **`SqliteSaver` wants an explicit connection for a long-lived graph.**
  `SqliteSaver.from_conn_string(path)` is a *context manager* — fine for a
  with-block, wrong for "build the graph, hand it around, resume later in
  another process." Construct `sqlite3.connect(path, check_same_thread=False)`
  and wrap it: `SqliteSaver(conn)`. `check_same_thread=False` because the saver
  may touch the connection from a worker thread. (phase 3)

- **Resuming is `invoke(None, config)`, and `interrupt_after` marks the pause.**
  Non-obvious from the docs: to *continue* a checkpointed run you invoke the
  graph with input `None` and the same `thread_id` — passing the state back in
  would restart it. To stop at a clean boundary for the demo,
  `compile(interrupt_after=["act"])` halts right after the observation is
  persisted, with `get_state(config).next` showing the node it will resume
  into. (phase 3)
