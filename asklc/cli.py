"""The asklc CLI — the same two verbs as askrepo, so the demos line up.

    secrun python -m asklc index ../DeepDives/deep-dive-capstone/fixtures
    secrun python -m asklc ask "how do you run Nimbus?"

Keyless (PROVIDER=mock, the default) both verbs run with no key and no network:
`index` embeds with a deterministic fake embedder, `ask` answers with a canned
mock that names itself. Retrieved chunks print to stderr either way, so even
the keyless path shows real retrieval — the plumbing is never a black box.
"""

import argparse
import sys

from asklc.config import load_config


def cmd_index(args):
    from asklc.rag import CHROMA_DIR, build_index

    stack = load_config()["PROVIDER"]
    if stack == "mock":
        print("provider: mock — embedding with a DETERMINISTIC FAKE embedder "
              "(no key, no network, not semantic).", file=sys.stderr)
    n_files, n_chunks = build_index(args.path, stack)
    print(f"indexed {n_files} files -> {n_chunks} chunks (stack: {stack})",
          file=sys.stderr)
    print(f"saved: {CHROMA_DIR}", file=sys.stderr)
    return 0


def cmd_ask(args):
    from asklc.rag import answer

    config = load_config()
    provider = config["PROVIDER"]
    print(f"provider: {provider}"
          + (" (canned mock answer — set PROVIDER=claude|openai for a real one)"
             if provider == "mock" else ""),
          file=sys.stderr)

    text, docs = answer(args.question, k=args.k)
    for d in docs:
        print(f"retrieved: {d.metadata.get('source','?')}:"
              f"{d.metadata.get('start_line','?')}-{d.metadata.get('end_line','?')}",
              file=sys.stderr)
    print(text, flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="asklc",
        description="Ask questions about a codebase — the LangChain port of askrepo.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="chunk + embed a corpus directory")
    index.add_argument("path", help="root of the corpus to index")
    index.set_defaults(func=cmd_index)

    ask = subparsers.add_parser("ask", help="retrieve, answer, cite")
    ask.add_argument("question", help="the question, in plain English")
    ask.add_argument("--k", type=int, default=None,
                     help="chunks to retrieve (default: K from config, 5)")
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)
