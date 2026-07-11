"""The prompt contract — ported verbatim from askrepo/prompts.py.

This is deliberately a copy, not an improvement. Phase 2 measures eval parity
between the two implementations; if the prompts differed, a score gap could be
the prompt talking, not the framework. So the system contract, the few-shots,
the decline phrase and the citation format are identical to askrepo's — the
only change is packaging them as a LangChain `ChatPromptTemplate` instead of a
hand-built message list.

The three rules the contract enforces (see askrepo/prompts.py for the why):
  1. Answer ONLY from provided context.
  2. Cite (path:line) for every claim.
  3. If the answer isn't in context, reply with DECLINE_PHRASE verbatim.
"""

from langchain_core.prompts import ChatPromptTemplate

# Kept in lockstep with askrepo/prompts.py — bump both together.
CONTRACT_VERSION = "2"
DECLINE_PHRASE = "Not in this corpus."
CITATION_FORMAT = "(path:line)"

SYSTEM_PROMPT = f"""\
You are askrepo, a codebase Q&A assistant. You answer questions about a code
repository using ONLY the context blocks provided in the conversation.

Rules — these override everything else:

1. Ground every claim in the provided context. Do not use prior knowledge,
   even when you are confident. If the context and your prior knowledge
   disagree, the context wins.
2. Cite the source of every claim as {CITATION_FORMAT}, e.g. (src/app.py:42)
   or (README.md:10-14), using the path and line numbers exactly as they
   appear in the context blocks. Every factual sentence needs at least one
   citation.
3. If the provided context does not contain the answer — or no context was
   provided at all — reply with exactly: {DECLINE_PHRASE} You may add one
   sentence suggesting where the answer might live, clearly marked as a
   guess. Never improvise an answer to be helpful.
4. Answer directly and concisely. No preamble like "Based on the context"."""

# Same two few-shots as askrepo: one grounded-and-cited answer, one decline.
FEW_SHOTS = [
    (
        "human",
        '<context path="tools/fmt.py">\n'
        "1| MAX_WIDTH = 88\n"
        "2|\n"
        "3| def wrap(text):\n"
        '4|     """Wrap text to MAX_WIDTH columns."""\n'
        "5|     return textwrap.fill(text, MAX_WIDTH)\n"
        "</context>\n\n"
        "Question: What line width does the formatter use?",
    ),
    (
        "ai",
        "The formatter wraps text to 88 columns — `MAX_WIDTH = 88` "
        "(tools/fmt.py:1), used by `wrap()` (tools/fmt.py:5).",
    ),
    (
        "human",
        '<context path="tools/fmt.py">\n'
        "1| MAX_WIDTH = 88\n"
        "</context>\n\n"
        "Question: Which linter does this project use?",
    ),
    (
        "ai",
        f"{DECLINE_PHRASE} (Guess: a linter would likely be configured in "
        "pyproject.toml or a CI workflow, which aren't in the provided "
        "context.)",
    ),
]


def format_context(path, text, start=1):
    """One line-numbered context block, identical shape to askrepo's.

    `start` is the chunk's first line in the original file (from
    corpus.chunk_documents), so a citation points where the chunk actually
    lives, not at line 1.
    """
    numbered = "\n".join(
        f"{i}| {line}" for i, line in enumerate(text.splitlines(), start=start)
    )
    return f'<context path="{path}">\n{numbered}\n</context>'


# The template: system contract, the few-shots, then the real turn. `{context}`
# is the already-formatted, joined context blocks; `{question}` the user's ask.
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        *FEW_SHOTS,
        ("human", "{context}\n\nQuestion: {question}"),
    ]
)
