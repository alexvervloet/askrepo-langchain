# LESSONS

Engineering lessons from rebuilding [askrepo](https://github.com/alexvervloet/ask-my-repo)
— a from-scratch repo-Q&A tool with a hand-rolled chunker, index, and agent loop
— on **LangChain + LangGraph**, and measuring the two against each other. This is
a comparison project, so most of these are about the same discipline: **not
mistaking what the framework does for what you would have chosen, and not
mistaking a matched score for a matched system.** Each is tied to the concrete
thing that taught it. Kept as a running log; it grows as the phases land.

---

## 1. A framework's defaults are decisions someone else made for you — usually invisibly, sometimes wrong

Porting the RAG core meant inheriting a stack of defaults that the from-scratch
version had made explicit on purpose, and almost none of them matched. Chroma
ranks by **L2 distance, not cosine** — a default buried in chromadb's HNSW
config that `langchain-chroma` never surfaces, and it changes the *ranking*, not
just the score numbers, so an unconfigured store quietly retrieves different
neighbors than askrepo's cosine. The retriever returns **k=4**, not askrepo's
explicit 5 (`langchain_chroma.DEFAULT_K`). `RecursiveCharacterTextSplitter`
tracks **no position at all** unless you pass `add_start_index=True`, so
`(path:line)` citations are impossible out of the box — and when you enable it,
what you get is a *character offset*, not a line, so recovering the citation is
extra work the line-tracking hand-rolled chunker never needed. Every one of
these came from reading the installed source, not the component's docs.

Takeaway: adopting a framework means adopting its defaults as decisions, and the
docs describe the happy path, not what the component actually does. Before you
trust a default, read the source for the number — especially for anything that
silently changes results rather than failing loudly.

## 2. A matched aggregate is a hypothesis, not a conclusion — read the per-item view

The phase-2 eval looked like a dead tie: judged correctness 0.814 (from-scratch)
vs 0.800 (framework), and retrieval hit@k **identical to three decimals** at
0.857. Read as a headline, "the stock vector pipeline matches the hand-tuned
hybrid." But the identical hit@k is a *coincidence of offsetting differences*:
**12 of 40 questions behave differently**, and the wins cancel. askrepo's BM25
hybrid wins `code-07` (the vector-only retriever returned the *same README chunk
five times* and declined a question the keyword signal answered); the framework's
larger character-window chunks win `code-03` (they kept a whole function together
so the answer line was in the retrieved chunk, where askrepo's per-`def` chunker
had split it off). Two mechanisms pulling opposite directions, netting to a tie
that describes neither system.

Takeaway: when two systems score the same in aggregate, that is the beginning of
the analysis, not the end. Diff the per-item results; a matched mean often hides
large, systematic, offsetting differences, and the mechanism behind the churn is
the actual finding.

## 3. For an honest A/B, hold the measuring apparatus physically constant

The capstone ships a frozen `baseline.run.json`, and the lazy move was to compare
the new implementation's numbers against it. I didn't, because that baseline was
captured under a possibly different judge model and a different corpus SHA — any
gap could be the grader or the corpus moving, not the pipeline. Instead
`evals/parity.py` runs **both** pipelines back-to-back in one process, over the
same corpus snapshot, graded by **one shared gpt-4o-mini judge imported from the
capstone**, so the two columns cannot drift on anything except the thing under
test. The only variable left is the pipeline.

Takeaway: a comparison is only as trustworthy as what it holds constant. Don't
score today's system against yesterday's saved numbers; run both arms through the
same apparatus at the same time, and reuse the grader rather than re-implementing
it, so a difference can't hide in the measurement.

## 4. Control a comparison by *reusing* the parts you aren't measuring, not re-implementing them

Two ports could have quietly poisoned their own results. The prompt contract was
copied **verbatim** from askrepo into a `ChatPromptTemplate`, so a correctness
delta in phase 2 couldn't be the prompt talking. The LangGraph agent reuses
askrepo's **exact tools and read-only harness** (`grep`/`read_file`/`list_dir`
behind the same sandbox + policy + audit), so the phase-3 comparison is purely
about who runs the loop — a hand-written `while` vs a compiled graph — and not
about the tools or the security boundary, which are byte-identical. Re-writing
either would have introduced a second variable and made every difference
ambiguous.

Takeaway: hold your controlled variables identical by reuse, not by careful
re-implementation. Every line you rewrite "to match" is a line that can silently
diverge; importing the original guarantees the only thing that changed is the
thing you meant to change.

## 5. The capability that isn't in the box is the one worth checking survives the failure

LangGraph's headline advantage over the hand-rolled loop is durable state — kill
the process mid-task and resume. But `pip install langgraph` gives you
`MemorySaver`, which is **in-memory**: it demonstrates the *shape* of
checkpointing while being unable to survive the exact event (a process death)
that the feature exists for. Real cross-process resume needs a **separate
package**, `langgraph-checkpoint-sqlite`. Only after installing it could I prove
the claim honestly — two genuinely separate processes (pids 8210 → 8212), where
process 1 runs a real tool call, checkpoints four messages to SQLite, and exits
mid-task, and process 2 with an empty memory and a fresh model recovers the state
from disk and finishes. The hand-rolled loop keeps `messages` in a local
variable and structurally cannot do this at any price.

Takeaway: when you adopt a framework *for* a capability, test that the capability
survives the failure it's advertised against, not just that the API exists. The
default implementation often shows the shape without the substance, and the
substance may be a dependency you haven't installed yet.

## 6. The "dumber" default sometimes wins — measure chunking, don't reason about it

I expected askrepo's structure-aware chunker (one chunk per markdown heading or
python `def`/`class`, line-tracked) to clearly beat
`RecursiveCharacterTextSplitter`'s generic 1500-character windows — it's more
"intelligent." Over the same corpus it produced **2389 chunks vs the splitter's
1847**, and I'd have bet on the finer ones. The eval disagreed: they roughly
tied, and on `code-03` the coarse splitter *won*, because a 1500-char window
happened to keep an entire small function — guard clause and all — in one
retrievable unit, while the structural chunker split the answer line into a
neighboring chunk that never got retrieved. Structure-awareness bought more,
smaller, cleaner chunks; it did not buy accuracy.

Takeaway: intuition about which design is "smarter" is not evidence about which
retrieves better. A dumber, coarser default can win for a concrete mechanical
reason (keeping a whole unit together) that has nothing to do with elegance —
measure the two on your own corpus before believing the sophisticated one.

## 7. Framework velocity is a cost you pay while building, not just later

The churn showed up mid-project, repeatedly. The canonical loader for the `load`
stage, `TextLoader`, lives in `langchain-community`, which prints a **"being
sunset / no longer actively maintained"** DeprecationWarning on import — reading
a text file the framework way pulls in a package its own maintainers are winding
down, plus `langchain-classic` and sqlalchemy as transitive deps. The durable
checkpointer is a package split off from `langgraph` itself. None of this is
visible until you install and run; it's the tax of building on a fast-moving
ecosystem, and it lands as real dependency decisions in `requirements.txt`.

Takeaway: a framework's release velocity is part of its total cost of ownership.
The "canonical way" to do something can be deprecated the week you adopt it, and
capabilities migrate between packages — budget for the churn, and pin and record
the versions you measured against, because they *will* move.

## 8. Name precisely what the bought tool adds — the honest diff is narrower than the pitch

The easy pitch for LangSmith is "observability the hand-rolled tracer doesn't
have." Traced one RAG run and it captured **all 9 runnables** of the LCEL chain
automatically, nested, with per-step inputs and outputs — and that granularity
surfaced a real finding the coarse tracer hides: the vector **retriever
(1819 ms)**, dominated by the OpenAI query-embedding round-trip, is *slower than
the LLM call (1318 ms)*. But the honest comparison is narrower than the pitch:
askrepo's tracer **already** records total tokens and cost, as flat trace
attributes. So the bought tool's real additions are the automatic **nested
structure**, the **per-step I/O capture**, and **persistence/shareability** — not
the token and cost totals both tools have. Getting that boundary right matters
more than the sales line; the hand-rolled tracer still wins on zero dependencies
and no data leaving the box.

Takeaway: when you write up what a bought tool gives you, state the differentiator
exactly and subtract what the hand-rolled version already did. "It shows tokens
and cost" was false as a *difference*; "it shows structure and I/O, and keeps
them" was true. The credibility of a comparison lives in that precision.

## 9. An abstraction removes the seams you sometimes need to see

The framework's `Embeddings.embed_documents` returns only vectors. askrepo's
`embed()` returns `(vectors, tokens)`, and it uses that token count to print a
priced line on every index build — "indexed N files → M chunks (T embedding
tokens, $X)." That line simply cannot be reproduced through the LangChain path
without bypassing the abstraction and re-tokenizing the corpus myself. The
one-call convenience of `Chroma.from_documents(...)` is real, but it swallowed a
seam the from-scratch version deliberately kept open for cost visibility.

Takeaway: abstractions are subtractive as well as additive — every seam they hide
is a measurement or an intervention point you lose. When you count what a
framework buys in fewer lines, also count what it costs in visibility, and check
that the seams you actually need are still reachable.
