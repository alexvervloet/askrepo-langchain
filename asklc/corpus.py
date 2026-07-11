"""Load and chunk the corpus — the `load` and `chunk` stages, on LangChain.

askrepo hand-rolls both (indexer.py):

  - load:  os.walk with an extension allow-list and a skip-dir set, reading
           each file into a dict {path, text}.
  - chunk: two *structure-aware* splitters (markdown-by-heading,
           python-by-def/class), each tracking (start_line, end_line) so a
           citation resolves to a real place in a real file.

The framework's answer to both is a one-liner each: `TextLoader` for load,
`RecursiveCharacterTextSplitter` for chunk. This module uses exactly those,
and the docstrings mark every place a default silently replaces a decision
askrepo made on purpose — that delta is the whole point of the port.

Two deltas worth stating up front, because they change what the citations can
even say:

  1. RecursiveCharacterTextSplitter splits by *character count* against a
     generic separator ladder ("\\n\\n", "\\n", " ", ""), not by markdown
     headings or python objects. So a chunk can straddle two functions or cut a
     heading off its section — the structural guarantee askrepo pays for by
     hand is simply not in the box.
  2. The splitter tracks a character *offset* (metadata["start_index"], and
     only if you opt in with add_start_index=True — it defaults to False), not
     a line number. askrepo's citations are (path:line); to keep parity we
     convert the offset back to a line here, using the parent document's text.
     That conversion is work the hand-rolled chunker never needed — it tracked
     lines from the start.
"""

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Byte-for-byte the same walk policy as askrepo/indexer.py, so both
# implementations see the *identical* corpus — a difference downstream is then
# provably the framework's, not a different set of input files.
INDEXED_EXTENSIONS = {".md", ".py"}
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".history", ".vscode",
    "node_modules", "index", ".idea", ".chroma",
}

# askrepo makes these explicit (MAX_CHUNK_LINES=60, OVERLAP_LINES=10) in *lines*.
# The framework splitter counts *characters*, and its own defaults are
# chunk_size=4000, chunk_overlap=200 (langchain_text_splitters/base.py) — big
# enough that every file in a small corpus becomes a single unsplit chunk. We
# pick smaller, comparable character budgets so retrieval has something to rank,
# and record the mismatch (lines vs chars, 60/10 vs default 4000/200) in the
# mapping table rather than hiding it behind a matching number.
CHUNK_SIZE = 1500     # chars ≈ askrepo's 60-line cap for typical code/prose
CHUNK_OVERLAP = 200   # chars ≈ askrepo's 10-line window overlap


def load_documents(corpus_root):
    """Walk the corpus and load every .md/.py file as a LangChain Document.

    Uses LangChain's `TextLoader` per file — a real framework loader — but the
    *walk* (which files, which dirs to skip) is still ours: `DirectoryLoader`
    exists, but wiring its glob + exclude to match askrepo's exact allow-list is
    more fighting than just calling os.walk, and this way the corpus is provably
    identical. Metadata["source"] is the path *relative* to the corpus root, so
    citations read the same as askrepo's (README.md:10, not /abs/path/README.md).
    """
    corpus_root = os.path.abspath(corpus_root)
    docs = []
    for dirpath, dirnames, filenames in os.walk(corpus_root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if os.path.splitext(name)[1] not in INDEXED_EXTENSIONS:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, corpus_root)
            try:
                # autodetect_encoding keeps a stray non-utf8 file from aborting
                # the whole load — askrepo's loader just skips such files.
                loaded = TextLoader(full, autodetect_encoding=True).load()
            except (UnicodeDecodeError, OSError, RuntimeError):
                continue
            for d in loaded:
                d.metadata["source"] = rel
                docs.append(d)
    return docs


def _line_of(text, char_index):
    """1-based line number of a character offset in `text` (for citations)."""
    return text.count("\n", 0, char_index) + 1


def chunk_documents(docs):
    """Split Documents with RecursiveCharacterTextSplitter, recovering lines.

    add_start_index=True is the opt-in that makes the splitter record where each
    chunk began (as a *character* offset). We convert that offset to a 1-based
    line against the parent document and stash (start_line, end_line) in
    metadata, so downstream citation code matches askrepo's (path:line) format.
    Everything after the splitter call exists only because the framework tracks
    offsets, not lines.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # defaults to False — no start tracking at all
    )
    out = []
    # Split per-document so the parent text is in hand for the offset->line map.
    for doc in docs:
        for chunk in splitter.split_documents([doc]):
            start = chunk.metadata.get("start_index", 0)
            start_line = _line_of(doc.page_content, start)
            end_line = start_line + chunk.page_content.count("\n")
            chunk.metadata["start_line"] = start_line
            chunk.metadata["end_line"] = end_line
            out.append(chunk)
    return out
