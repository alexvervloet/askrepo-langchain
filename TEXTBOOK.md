# The Same App, Built on the Framework: What LangChain Buys and Costs

*This is the textbook companion for the askrepo-langchain project. The [README](README.md) tells you how to run it; [COMPARISON.md](COMPARISON.md) is the standalone verdict with the full tables; [PLAN.md](PLAN.md) has the per-phase detail and the running gotcha log; [LESSONS.md](LESSONS.md) collects the engineering takeaways. This piece is the lecture: what it means to rebuild a hand-written application on a framework, what the framework genuinely gives you, where it quietly works against you, and how to read an evaluation that looks like a tie but is not one. It assumes you know roughly what LangChain and LangGraph are (popular frameworks for building applications on language models) and want an honest accounting of the tradeoff.*

---

## 1. The question the job market keeps asking

"LangChain" appears in a large fraction of AI-engineering job descriptions, and LangGraph, its graph-based cousin for building stateful agents, is the part production teams actually run. So there is a real, career-relevant question underneath the hype: what does a framework like this actually buy you, and what does it cost, versus code you write and control yourself? The answer people usually give is a vibe ("frameworks are bloated" or "frameworks save time"), and a vibe is not evidence.

This project turns the question into an experiment. There is an existing application, askrepo, that answers questions about a code repository with grounded, cited answers: ask "where is retrieval implemented?" and it replies from the actual files, tagging each claim with its `(path:line)` location, or declines with "Not in this corpus." when the answer is not there. That application was built entirely from scratch: a hand-written chunker, its own index format, a hybrid retriever that blends vector and keyword search, and a hand-rolled agent loop, no framework anywhere. This project rebuilds the identical application on LangChain and LangGraph, feeds both versions the byte-identical corpus and the same 40 gold questions, and measures them against each other with a single shared judge.

The design choice that makes this worth doing is that the from-scratch version came first. Because every piece already exists as hand-written code, each framework component maps to something concrete you can point at, which makes the framework legible instead of magical. And when the two versions disagree (different chunks, different retrieval, different eval scores), the *disagreement* is the finding, because it isolates exactly what the framework changed.

## 2. The lines-of-code story is a trap

The obvious first measurement is how much code each version takes, stage by stage. Here it is, functional lines only:

| Stage | From scratch | Framework | The headline difference |
|---|---|---|---|
| load | ~30 | ~25 | the stock loader needs a package that is being sunset; the directory walk is still yours either way |
| chunk | ~55 | ~25 | characters, not lines, so no `(path:line)` citations unless you opt into position tracking |
| embed | ~25 | ~6 | one call instead of a batch loop, but you cannot price the build; the token count is hidden |
| store | ~15 | ~13 | the default vector store ranks by Euclidean distance, not cosine, silently changing rankings |
| retrieve | ~90 | ~2 | the 90 lines bought a hybrid retriever; the stock one is vector-only |
| answer | ~50 | ~30 | the framework's chain syntax is tidy; token-budgeted assembly is still do-it-yourself |

At a glance this looks like a clear framework win: retrieval collapses from 90 lines to 2. But that collapse is exactly the trap, and seeing why is the most important idea in the whole comparison. Those 90 lines *were* the vector-plus-keyword hybrid retriever, and the evaluation below shows that hybrid earns real, measurable wins. The two-line replacement is a vector-only retriever that does less. Meanwhile, the stages that barely shrank (load and chunk) stayed large because the directory walk, the recovery of line numbers for citations, and the override of the wrong default distance metric all had to be written back by hand.

So the "framework writes less code" claim holds at exactly one stage, and it is the stage where writing less code also means *doing less*, and where doing less measurably costs you retrieval quality. A lines-of-code count, taken at face value, would have told you the framework is a big win. Read against what each stage actually does, it tells a subtler and more honest story: the framework saves the most typing precisely where you least want it to.

## 3. What the framework genuinely bought

An honest comparison has to give the framework its real wins, and there are several, concentrated in one area.

The strongest result in the entire project is **durable agent state**, and it is a genuine capability rather than a convenience. The from-scratch agent keeps its conversation in an ordinary in-memory variable; kill the process and the task is gone. The LangGraph version checkpoints its state after every step, and with a database-backed saver that state survives the process ending entirely. This was proven with two separate processes: the first runs a real tool call, checkpoints, and exits mid-task; the second, a different process with empty memory and a freshly loaded model, recovers the state from disk and finishes the job. This is not a lines-of-code saving. It is a capability that the hand-rolled loop structurally cannot have without building a serializer by hand, and for anything stateful (an agent that must survive a crash, a long tool loop, a resumable job) it is exactly the kind of thing you do not want to build yourself.

Attached to that runtime came more: per-node retry, timeout, and cache policies as simple configuration arguments, where the from-scratch version hand-wrote a retry wrapper and applied it in only one place. The whole retrieval pipeline is expressible as a single composed chain, so swapping the vector store, the embedder, or the chat model is a one-object change rather than an edit across hand-written provider classes. And there was free plumbing that the from-scratch version either wrote by hand or simply lacks: automatic query-type handling for certain embedding providers, node-level progress streaming, and the ability to travel back through the history of checkpoints.

The framework also bought observability. With one environment variable and no instrumentation of the project's own, the framework's tracing platform captured every step of the retrieval chain, nested, with per-step inputs and outputs, token counts, and automatically-priced cost, persisted and shareable. It even revealed something the hand-written version hid: the retriever (about 1,819 milliseconds) actually outweighed the language-model call (about 1,318 milliseconds), a split that the from-scratch version's two coarse hand-wrapped spans could not show. The hand-written tracer still wins on having zero dependencies and keeping all data on the machine; the framework's platform wins on structure, on capturing inputs and outputs, and on persistence. That is a real tradeoff, not a rout in either direction.

## 4. What the framework quietly cost

The costs are subtler than the wins, and nearly all of them share a theme: the framework makes decisions for you, invisibly, and often not the decision you would have made.

The clearest example is **silent defaults that were wrong for the task and could only be found by reading the source**. The default vector store ranks results by Euclidean distance rather than cosine similarity, which changes which chunks come back, not merely the numbers attached to them. The default retriever returns four results, not five. The default text splitter tracks no position information at all unless you opt in, and even then it tracks a character offset rather than a line number, which is useless for `(path:line)` citations without extra work. Every one of these came from reading the installed source code, not from the documentation, and every one of them would have silently degraded the application if left alone.

There was **no hybrid retrieval available by default**. The from-scratch version's blend of keyword and vector search has no stock analogue; the default retriever is vector-only, and on one specific question it returned the same README chunk five times and declined a question the hand-written version answered correctly. Matching the hybrid means wiring together several framework components yourself, at which point the "less code" advantage has evaporated.

There was **version churn during the project itself**. The canonical loader lives in a package that prints a "being sunset, no longer maintained" warning on import. A durable checkpointer is a separate package from the core one you install. You feel the ecosystem shifting underneath you while you are building on it, which is a real cost that does not show up in any feature comparison.

And there was **lost visibility**. The framework's embedding interface returns only vectors, so the per-build token-and-cost line that the from-scratch version prints cannot be reproduced without bypassing the abstraction and re-tokenizing the text yourself. The abstraction that made embedding a six-line call also hid the information you would need to budget an index build.

## 5. The evaluation that looks like a tie and is not

Here is where the project earns its keep. Both versions were run over all 40 gold questions with a single shared judge, the from-scratch version using its real hybrid retriever and the framework version using its stock vector-only retriever, because vector-only is precisely the default being measured.

| Metric | From scratch | Framework |
|---|---|---|
| judged correctness (32 answerable) | 0.814 | 0.800 |
| decline accuracy (8 negatives) | 1.000 | 1.000 |
| retrieval hit rate (an expected file in the top k) | 0.857 | 0.857 |
| tokens per answer, median | 2,546 | 2,419 |
| citation resolve / match | 0.881 / 0.667 | 0.951 / 0.732 |

At first glance this is a tie, and a naive reading would conclude the framework's stock retriever is just as good as the hand-tuned hybrid. The retrieval hit rate is identical to three decimal places. But the tie is *coincidental*, and that is the actual finding. Twelve of the forty questions behave differently between the two versions; the differences happen to cancel out in the aggregate. Two mechanisms pull in opposite directions.

The from-scratch version's keyword-plus-vector hybrid wins when the answer sits in a specific file named by exact terms. On one question about when an agent loop stops, the framework's vector-only retriever returned the same README chunk five times and declined, while the hybrid's keyword signal surfaced the exact line. Pulling the other way, the framework's larger character-window chunks win when the answer is a single line inside a function that the from-scratch version's per-function chunker splits apart. On a question about a specific function's behavior, the framework's large chunk kept the whole function together and cited it, while the from-scratch version retrieved a different slice and declined.

So a stock vector retriever *matched* a hand-tuned hybrid on this corpus, but by trading wins question-for-question, not because the two pipelines are equivalent. The framework version even edged citation quality, because its bigger chunks carry a wider, correctly-numbered line span, so a cited line lands inside a real chunk more often. The correctness gap of 0.814 versus 0.800 is well within the judge's own run-to-run nondeterminism; the real signal is the churn and its two named causes.

The practitioner lesson is sharp: the framework's default retriever has two traps this evaluation surfaced, and neither is on by default, and the aggregate score will not warn you about either. It returns near-duplicate chunks with no diversity, fixable by switching to a retrieval mode that spreads results out, and it has no keyword channel for exact-match terms, fixable by adding a keyword retriever alongside the vector one. **Only the per-question view revealed any of this.** A matched aggregate hid twelve questions' worth of offsetting differences, which is a general warning about trusting a single summary number to tell you two systems are the same.

## 6. The verdict is about layers, not frameworks

The honest conclusion is not "use a framework" or "don't." It is that the two layers of this application have opposite answers.

For the **agent runtime**, reach for the framework. Specifically, LangGraph's durable persistence and cross-process resume is the single strongest result in the whole comparison: a real capability the hand-rolled loop cannot have without building a serializer, and one that arrived with per-node retries, streaming, and checkpoint history attached. If you are shipping anything stateful (an agent that must survive a crash, a long tool loop, a resumable job), that is where the abstraction is buying capability rather than just syntax, and it is the part production teams actually run.

For the **retrieval core**, keep it thin and explicit, whether hand-rolled or built on a minimal vector library. Porting the retrieval layer to the framework did not buy accuracy; it tied. And it cost the hybrid retrieval that measurably wins questions, the ability to price an index build, and a stack of silent defaults you have to discover by reading source. When you care about specific retrieval behavior (hybrid search, custom chunking, cost visibility), the framework's convenience works against you by hiding exactly the knobs you need. The framework's retrieval chain was the weakest value in this project; its runtime was the strongest.

And underneath both layers sits the lesson that outlasts any particular framework version: **the framework's defaults are decisions someone else made for you, often invisibly, and often not the decision you would have made.** Read the source for the numbers that matter, diff your evaluations per-question rather than trusting the aggregate, and never let a matched summary number convince you two systems are equivalent, because here it hid twelve questions' worth of real, offsetting differences. A framework is an excellent place to start and a poor place to stop reading.

---

*Run it: [README.md](README.md) · The full verdict with all tables: [COMPARISON.md](COMPARISON.md) · Per-phase detail: [PLAN.md](PLAN.md) · Engineering lessons: [LESSONS.md](LESSONS.md)*
