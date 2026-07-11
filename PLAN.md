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
| load | | | |
| chunk | | | |
| embed | | | |
| store/index | | | |
| retrieve | | | |
| assemble+answer | | | |

Questions to answer in writing: which knobs did askrepo make explicit that the
framework defaults silently (chunk size, overlap, separators, k, score
threshold)? Where did you have to read LangChain *source* rather than docs to
find out what a component actually does?

## Phase 2 — eval parity (done when both columns are filled)

Run the capstone's gold questions against both implementations.

| metric | askrepo | asklc (LangChain) |
|---|---|---|
| gold questions passed | | |
| avg retrieval hit (gold chunk in context) | | |
| tokens per answer (p50) | | |

If the scores differ, diagnose *why* (chunking? retrieval? prompt template?)
before moving on — the diagnosis is the most valuable paragraph in
COMPARISON.md.

## Phase 3 — agent on LangGraph (done when parity + resume both work)

Port the v05-agent tool loop to a `StateGraph`: explicit nodes for
plan/act/observe, tools bound to the model, checkpointer on.

- [ ] the agent completes the same tasks the capstone's hand-rolled loop does
- [ ] kill the process mid-run, restart, and **resume from the checkpoint** —
      the thing the hand-rolled loop can't do; demo it honestly
- [ ] capability table: loop control / retries / persistence / streaming —
      hand-rolled vs LangGraph, with what each cost to get

## Phase 4 — observability (stretch)

One traced run with LangSmith (or the local tracer) next to the observability
dive's hand-rolled approach: what a bought trace shows that the hand-rolled
one doesn't, in three sentences and a screenshot.

## Phase 5 — the write-up (done when COMPARISON.md stands alone)

- [ ] COMPARISON.md complete: **bought / cost / when to choose which**
- [ ] cross-link from the portfolio README / capstone LEARNINGS
- [ ] sanity check: could someone who never read the capstone follow it? fix
      until yes

## Notes / gotchas discovered along the way

(keep a running log here — this becomes the best part of COMPARISON.md)
