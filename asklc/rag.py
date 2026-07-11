"""The RAG pipeline on LangChain: embed -> store -> retrieve -> assemble+answer.

This is the heart of the phase-1 port. Each stage below has a one-to-one
counterpart in askrepo, called out so the mapping table writes itself:

  embed+store   askrepo/indexer.py build_index (embed batches, round vectors,
                json.dump)  ->  Chroma.from_documents (embeds + persists a
                collection). ~30 hand-rolled lines become one call.
  retrieve      askrepo/retrieve.py (cosine + BM25, min-max blended, k=5)  ->
                store.as_retriever(search_kwargs={"k": k}). One line — but
                *vector-only*: the framework retriever has no BM25 blend, so
                exact-match terms (module names, flags) lose the keyword signal
                askrepo blends in at 0.7. That's the biggest behavioural delta
                to explain in phase 2.
  assemble      askrepo/assemble.py (token-budgeted greedy keep)  ->  a plain
                join of the k blocks. The framework hands you k docs; a budget
                policy is yours to add if you want one (we don't need it at k=5).
  answer        askrepo/answer.py + providers.complete  ->  the LCEL chain
                below: retrieve | prompt | model | parse, as one expression.

Two silent Chroma defaults we override for a fair fight, both found by reading
source, not docs:
  - distance metric: Chroma's HNSW default is L2 (chromadb), askrepo uses
    cosine. We pass hnsw:space=cosine so both rank in the same geometry.
  - k: the retriever's default is 4 (langchain_chroma DEFAULT_K); askrepo's
    explicit default is 5. We pass k from config.
"""

import datetime
import json
import os
import shutil

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from asklc.config import CHROMA_DIR, COLLECTION, load_config
from asklc.corpus import chunk_documents, load_documents
from asklc.prompt import PROMPT, format_context
from asklc.providers import embed_model_name, get_chat_model, get_embeddings

META_PATH = os.path.join(CHROMA_DIR, "asklc-meta.json")


# --- embed + store -----------------------------------------------------------

def build_index(corpus_root, stack, extra_skip_dirs=frozenset()):
    """Load, chunk, embed and persist the corpus as a Chroma collection.

    Returns (n_files, n_chunks). askrepo returns token/cost too; LangChain's
    embeddings don't surface a token count through this path, which is itself a
    finding — the from-scratch version prices every index build, the framework
    one can't without reaching past the abstraction. Recorded, not faked.

    `extra_skip_dirs` is forwarded to the loader (phase 2 excludes the capstone
    repo so the corpus matches askrepo's exactly).
    """
    docs = load_documents(corpus_root, extra_skip_dirs=extra_skip_dirs)
    chunks = chunk_documents(docs)
    if not chunks:
        raise SystemExit(f"No .md or .py files found under {corpus_root!r}.")

    # Fresh build every time, like askrepo overwriting index.json — otherwise a
    # re-index appends duplicate vectors to the existing collection.
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(stack),
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        # override Chroma's L2 default so we rank by cosine, like askrepo
        collection_metadata={"hnsw:space": "cosine"},
    )

    n_files = len({c.metadata["source"] for c in chunks})
    _write_meta(stack, corpus_root, n_files, len(chunks))
    return n_files, len(chunks)


def _write_meta(stack, corpus_root, n_files, n_chunks):
    """Sidecar recording which stack embedded the index.

    askrepo stores the stack *inside* index.json and reads it back at query
    time, because a query must be embedded with the same model as the documents
    (vectors from different models live in different spaces). Chroma has no slot
    for that, so we keep a sidecar and enforce the same rule in load_store().
    """
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "corpus_root": os.path.abspath(corpus_root),
                "stack": stack,
                "embed_model": embed_model_name(stack),
                "n_files": n_files,
                "n_chunks": n_chunks,
            },
            f,
        )


# --- retrieve ----------------------------------------------------------------

def load_store():
    """Open the persisted collection, embedding queries with the index's stack.

    Returns (store, meta). The stack comes from the sidecar, not from PROVIDER,
    so you can answer with Claude over an OpenAI-embedded index without
    corrupting the query geometry — the same invariant askrepo enforces.
    """
    if not os.path.exists(META_PATH):
        raise SystemExit(
            "No index found. Build one first:\n"
            "    secrun python -m asklc index <corpus-path>"
        )
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(meta["stack"]),
        persist_directory=CHROMA_DIR,
    )
    return store, meta


def format_docs(docs):
    """Join retrieved docs into the line-numbered context the contract expects."""
    return "\n\n".join(
        format_context(
            d.metadata.get("source", "?"),
            d.page_content,
            start=d.metadata.get("start_line", 1),
        )
        for d in docs
    )


# --- assemble + answer (the LCEL chain) --------------------------------------

def make_chain(store, stack, k, model=None):
    """The whole RAG pipeline as one LCEL expression: retrieve -> prompt ->
    model -> parse. This is the artifact phase 1 exists to produce — askrepo's
    answer.prepare + providers.complete, expressed as a runnable graph."""
    retriever = store.as_retriever(search_kwargs={"k": k})
    return (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | PROMPT
        | get_chat_model(stack, model)
        | StrOutputParser()
    )


def answer(question, k=None):
    """Answer a question over the persisted index; return (text, sources).

    Retrieval runs once, explicitly, so the CLI can show which chunks grounded
    the answer (retrieval is never a black box — askrepo's promise). The answer
    half then runs as the LCEL chain prompt | model | parse over those chunks.
    """
    config = load_config()
    k = k if k is not None else int(config["K"])
    store, meta = load_store()

    retriever = store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)  # the one query embedding

    chain = PROMPT | get_chat_model(meta["stack"], config["MODEL"] or None) | StrOutputParser()
    text = chain.invoke({"context": format_docs(docs), "question": question})
    return text, docs
