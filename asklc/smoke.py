"""Keyless end-to-end smoke: prove the LangChain and LangGraph plumbing works
before any API key is involved. Fake models only — no network, no cost.

Two halves, matching the two halves of the project:
  1. an LCEL retrieval chain (phase 1's shape, on fake embeddings + fake chat)
  2. a LangGraph StateGraph with a checkpointer (phase 3's shape)

Run: python -m asklc.smoke
"""

from typing import TypedDict

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

BANNER = "=" * 62
print(BANNER)
print("  SMOKE RUN — FAKE MODELS ONLY (no key, no network, no cost)")
print("  Answers below are canned; this only proves the plumbing.")
print(BANNER)

# --- 1. LCEL retrieval chain -------------------------------------------------

DOCS = [
    "askrepo indexes a repository into chunks and answers questions about it.",
    "Retrieval finds the top-k chunks most similar to the question.",
    "The agent loop lets the model call tools until the task is done.",
]

store = InMemoryVectorStore.from_texts(DOCS, DeterministicFakeEmbedding(size=64))
retriever = store.as_retriever(search_kwargs={"k": 1})

prompt = ChatPromptTemplate.from_template(
    "Answer from context.\n\nContext: {context}\n\nQuestion: {question}"
)
llm = FakeListChatModel(responses=["[FAKE] retrieval-augmented answer"])

chain = (
    {
        "context": retriever | (lambda docs: docs[0].page_content),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("what does retrieval do?")
assert answer == "[FAKE] retrieval-augmented answer"
print("\n✓ LCEL chain: retrieve → prompt → chat → parse ran end to end")
print(f"    answer: {answer}")

# --- 2. LangGraph graph with a checkpointer ----------------------------------


class State(TypedDict):
    question: str
    context: str
    answer: str


def retrieve(state: State) -> dict:
    docs = retriever.invoke(state["question"])
    return {"context": docs[0].page_content}


def respond(state: State) -> dict:
    msg = llm.invoke(f"{state['context']}\n\n{state['question']}")
    return {"answer": msg.content}


graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("respond", respond)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "respond")
graph.add_edge("respond", END)

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "smoke"}}
result = app.invoke({"question": "what is the agent loop?"}, config)

assert result["answer"].startswith("[FAKE]")
snapshot = app.get_state(config)
assert snapshot.values["answer"] == result["answer"]
print("\n✓ LangGraph: retrieve → respond graph ran; state is in the checkpointer")
print(f"    checkpointed answer: {snapshot.values['answer']}")

print(f"\n{BANNER}\nAll plumbing works. Phase 0 smoke: PASS\n{BANNER}")
