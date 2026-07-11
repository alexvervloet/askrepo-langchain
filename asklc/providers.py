"""Provider selection — the `embed` and `answer` model backends.

askrepo/providers.py hand-writes one class per provider (Mock/OpenAI/Claude/
Local), each honoring a `complete(messages) -> stream` contract, plus an
`embed(texts, stack)` function. The port's job is smaller: LangChain already
ships a `BaseChatModel` and an `Embeddings` interface, so "swap the provider"
is "return a different object." This module is where that swap lives.

Same stack contract as askrepo (PROVIDER=mock|claude|openai):

  claude  -> Voyage embeddings (voyage-3.5) + ChatAnthropic (claude-haiku-4-5)
  openai  -> OpenAI embeddings (text-embedding-3-small) + ChatOpenAI (gpt-4o-mini)
  mock    -> DeterministicFakeEmbedding + FakeListChatModel — KEYLESS, and it
             says so loudly. Same rule as every repo in this series: the
             keyless path never silently pretends a real model ran.

The models are identical to askrepo's so phase-2 eval parity compares
frameworks, not model choices. Anthropic ships no first-party embeddings, so
the claude stack embeds with Voyage — exactly the seam askrepo documents.
"""

from asklc.config import load_config

# Same model ids and dimensions as askrepo/providers.py, so an index built by
# either implementation embeds in the *same* vector space.
EMBED_MODELS = {
    "openai": "text-embedding-3-small",
    "claude": "voyage-3.5",
}
CHAT_MODELS = {
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5",
}
VOYAGE_DIM = 1024  # voyage-3.5's dimensionality; the fake embedder matches it
MAX_TOKENS = 1024  # single-question answers, same cap as askrepo


def embed_model_name(stack):
    """The embedding model id an index built on `stack` used (for the sidecar)."""
    if stack == "mock":
        return f"fake-deterministic-{VOYAGE_DIM}"
    return EMBED_MODELS[stack]


def get_embeddings(stack):
    """The LangChain Embeddings object for a stack.

    The query-vs-document `input_type` distinction askrepo passes by hand
    (embed(..., input_type="query")) is *automatic* here: VoyageAIEmbeddings
    calls embed_documents with "document" and embed_query with "query" for you
    (langchain_voyageai/embeddings.py). A convenience that happens to match —
    but one you only learn by reading the source, since nothing in the
    retriever API surfaces it.
    """
    if stack == "mock":
        # Deterministic so a keyless index is reproducible run to run. Same
        # width as voyage-3.5 purely for realism — it never leaves the process.
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=VOYAGE_DIM)
    if stack == "claude":
        from langchain_voyageai import VoyageAIEmbeddings

        # batch_size: askrepo sends 100/request; Voyage's LangChain default is
        # 1000. Doesn't change the vectors, only the request count — noted, not
        # matched. truncation defaults to True here (askrepo never truncates).
        return VoyageAIEmbeddings(model=EMBED_MODELS["claude"], batch_size=100)
    if stack == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBED_MODELS["openai"])
    raise SystemExit(
        f"PROVIDER={stack!r} has no embedding stack. Use mock, claude, or openai."
    )


def get_chat_model(stack, model=None):
    """The LangChain chat model for a stack (mock is keyless, canned)."""
    if stack == "mock":
        from langchain_core.language_models import FakeListChatModel

        # Honest canned answer: names itself a mock so plumbing is never
        # mistaken for intelligence. The CLI still prints the *real* retrieved
        # chunks to stderr, so retrieval is exercised and visible even keyless.
        return FakeListChatModel(
            responses=[
                "[mock] No model was called and no key was needed — this canned "
                "answer proves the LCEL chain ran: retrieve -> prompt -> chat -> "
                "parse. Set PROVIDER=claude or PROVIDER=openai for a real answer."
            ]
        )
    if stack == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model or CHAT_MODELS["claude"], max_tokens=MAX_TOKENS)
    if stack == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or CHAT_MODELS["openai"], max_tokens=MAX_TOKENS)
    raise SystemExit(
        f"PROVIDER={stack!r} is not recognized. Use mock, claude, or openai."
    )


def resolve_stack():
    """The active stack from config (PROVIDER), lowercased."""
    return load_config()["PROVIDER"]
