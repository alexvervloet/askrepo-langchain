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

*(evidence, not vibes: LOC deltas, components swapped with one line,
checkpoint/resume demo from phase 3, anything the hand-rolled version simply
doesn't have)*

## What the framework cost

*(evidence again: defaults that had to be hunted down in source, abstraction
layers crossed while debugging, version churn encountered during the project
itself, anything that was harder than the hand-rolled equivalent)*

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
