"""The agent loop on LangGraph — a StateGraph port of askrepo/agent.py.

askrepo's loop is a hand-written `while n_calls < MAX`: call the model, if it
asked for tools run them, feed results back, repeat (agent.py:237). This is the
same loop expressed as a graph so the framework owns the control flow and — the
part the hand-rolled version simply cannot do — the *state*, in a checkpointer
that survives the process dying.

The graph is the idiomatic two-node ReAct shape; the plan/act/observe the plan
asks for maps onto it like this:

    reason  (the model node)   plan + act: the model decides, and either emits
                               tool calls or a final answer
    act     (the tools node)   observe: run the requested tools, append results
    reason -> act -> reason … until the model answers or the budget is spent

**The tools and the boundary are askrepo's, reused verbatim.** Same grep /
read_file / list_dir, same ReadOnlySandbox + PermissionPolicy + AuditLog. That
is deliberate: if the tools differed, an agent-quality delta could be the tools
talking. Here the *only* thing that changed is who runs the loop — a hand-rolled
`while` vs a compiled graph with a checkpointer — which is exactly the phase-3
comparison.
"""

import sys
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Reuse the capstone's tools + harness so the boundary is byte-identical.
# Path resolves to a sibling checkout by default; override with ASKLC_CAPSTONE.
from asklc.config import CAPSTONE_ROOT as CAPSTONE


def _askrepo():
    """Lazily import askrepo's agent tools + harness (keeps asklc importable
    even where the capstone isn't checked out)."""
    if CAPSTONE not in sys.path:
        sys.path.insert(0, CAPSTONE)
    import askrepo.agent as A
    import askrepo.harness as H

    return A, H


MAX_TOOL_CALLS = 12  # same budget as askrepo/agent.py


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    n_calls: int


def make_tools(harness, touched):
    """askrepo's three read tools as LangChain tools, each closing over the
    harness so every call goes through policy -> sandbox -> audit exactly as in
    askrepo. `touched` accumulates the files the agent actually looked at — the
    agent-mode analogue of RAG's "retrieved" set."""
    A, _ = _askrepo()

    @tool
    def grep(pattern: str, path: Optional[str] = None) -> str:
        """Regex search across all .md and .py files (case-insensitive).
        Returns matching lines as path:line: text. Best first move."""
        return A.run_tool(harness, "grep", {"pattern": pattern, "path": path}, touched)

    @tool
    def read_file(path: str, start_line: int = 1) -> str:
        """Read up to 100 numbered lines of a file from start_line. Cite these
        line numbers."""
        return A.run_tool(
            harness, "read_file", {"path": path, "start_line": start_line}, touched
        )

    @tool
    def list_dir(path: str = ".") -> str:
        """List a directory: subdirectories (with /) and files."""
        return A.run_tool(harness, "list_dir", {"path": path}, touched)

    return [grep, read_file, list_dir]


def build_agent(model, harness, touched, checkpointer, interrupt_after=None):
    """Compile the ReAct graph: reason -> act -> reason, tools bound to `model`,
    state persisted in `checkpointer`.

    `model` is any LangChain chat model (real, or a scripted fake for the
    keyless resume demo). It is bound to the tools here; the tools node runs the
    same tool objects, so the schema the model sees and the code that executes
    are guaranteed to match.

    `interrupt_after` (e.g. ["act"]) halts the graph at a super-step boundary
    with the state checkpointed — the resume demo uses it to stop right after an
    observation is persisted, then continue in a *different process*.
    """
    A, _ = _askrepo()
    tools = make_tools(harness, touched)
    by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    def reason(state: AgentState) -> dict:
        # Budget spent: ask for a final answer with no tools (askrepo's
        # "Tool budget exhausted" turn). Otherwise let the model plan/act.
        if state["n_calls"] >= MAX_TOOL_CALLS:
            nudge = HumanMessage(
                content="Tool budget exhausted. Answer now from what you have "
                f"read, with citations — or reply with: {A.DECLINE_PHRASE}"
            )
            return {"messages": [model.invoke(state["messages"] + [nudge])]}
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def act(state: AgentState) -> dict:
        last = state["messages"][-1]
        outs = []
        for call in last.tool_calls:
            result = by_name[call["name"]].invoke(call["args"])
            outs.append(ToolMessage(content=result, tool_call_id=call["id"]))
        return {"messages": outs, "n_calls": state["n_calls"] + len(last.tool_calls)}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state["n_calls"] < MAX_TOOL_CALLS:
            return "act"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason)
    graph.add_node("act", act)
    graph.add_edge(START, "reason")
    graph.add_conditional_edges("reason", route, {"act": "act", END: END})
    graph.add_edge("act", "reason")
    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after or [])


def initial_state(question) -> AgentState:
    """The opening messages, using askrepo's agent system prompt verbatim."""
    A, _ = _askrepo()
    return {
        "messages": [
            SystemMessage(content=A.AGENT_SYSTEM),
            HumanMessage(content=f"Question: {question}"),
        ],
        "n_calls": 0,
    }


def answer(question, corpus_root, model, checkpointer, thread_id="agent"):
    """Run the agent to completion; return (text, touched, n_calls).

    A convenience wrapper for the real-model path. The resume demo drives the
    compiled graph directly (start in one process, resume in another) to prove
    the checkpoint survives a process boundary.
    """
    _, H = _askrepo()
    harness = H.default_harness(corpus_root)
    touched = set()
    app = build_agent(model, harness, touched, checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    final = app.invoke(initial_state(question), config)
    text = final["messages"][-1].content
    return text, sorted(touched), final["n_calls"]
