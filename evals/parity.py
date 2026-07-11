"""Phase-2 eval parity: the capstone's gold set against BOTH implementations,
back to back, in one process, with the SAME judge — so any score gap is the
framework, not the measurement.

    PROVIDER=openai secrun.sh .venv/bin/python evals/parity.py            # all 40
    PROVIDER=openai secrun.sh .venv/bin/python evals/parity.py --limit 3  # smoke

Design decisions that make this a fair A/B:

  - Same corpus. Both index the DeepDives series; asklc excludes the capstone
    repo exactly as askrepo excludes itself (extra_skip_dirs).
  - Same provider + k. Both answer with gpt-4o-mini, k=5. askrepo keeps its
    real config (hybrid retrieval, blend=0.7) because "the framework's default
    retriever is vector-only" is precisely the difference we're measuring — not
    something to paper over.
  - Same judge. One gpt-4o-mini judge grades both answers with the capstone's
    exact JUDGE_SYSTEM and scoring (score_citations, path_matches_any),
    imported from the capstone so grading can't drift between the two columns.
  - Same prompt contract. asklc ported askrepo's system prompt + few-shots
    verbatim (asklc/prompt.py), so a correctness delta isn't the prompt talking.

Real tokens for both: askrepo exposes provider.usage; asklc reads the
AIMessage.usage_metadata. No estimates.
"""

import argparse
import json
import os
import statistics
import sys
import time

# repo root first (so asklc.config resolves the sibling-checkout paths), then
# the capstone + its evals dir for the shared pipeline/grader imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asklc.config import CAPSTONE_ROOT as CAP  # noqa: E402
from asklc.config import CORPUS_ROOT as CORPUS  # noqa: E402

sys.path.insert(0, CAP)
sys.path.insert(0, os.path.join(CAP, "evals"))

# --- capstone measurement infrastructure (shared grader for both columns) ----
import run_evals as R  # noqa: E402  score_citations, path_matches_any, judge, JUDGE_SYSTEM

# --- askrepo pipeline (the from-scratch column) ------------------------------
from askrepo.answer import prepare as askrepo_prepare  # noqa: E402
from askrepo.prompts import DECLINE_PHRASE  # noqa: E402
from askrepo.providers import cost_usd, get_provider  # noqa: E402

# --- asklc pipeline (the framework column) -----------------------------------
from asklc.prompt import PROMPT  # noqa: E402
from asklc.providers import get_chat_model  # noqa: E402
from asklc.rag import format_docs, load_store  # noqa: E402

GOLDEN = os.path.join(CAP, "evals", "golden.jsonl")
K = 5


def run_askrepo(question, answer_provider):
    """askrepo's hybrid RAG: prepare (retrieve+assemble) -> complete."""
    messages, sources = askrepo_prepare(question, k=K, blend=0.7)
    t0 = time.perf_counter()
    text = "".join(answer_provider.complete(messages))
    latency = time.perf_counter() - t0
    in_tok, out_tok = answer_provider.usage
    retrieved = [c["path"] for _, c in sources]
    return text, retrieved, in_tok, out_tok, latency


def make_asklc_runner():
    """asklc's stock vector RAG: as_retriever(k) -> prompt | model."""
    store, meta = load_store()
    retriever = store.as_retriever(search_kwargs={"k": K})
    chain = PROMPT | get_chat_model("openai")  # returns the AIMessage (for usage)

    def run(question):
        t0 = time.perf_counter()
        docs = retriever.invoke(question)
        msg = chain.invoke({"context": format_docs(docs), "question": question})
        latency = time.perf_counter() - t0
        u = msg.usage_metadata or {}
        retrieved = [d.metadata.get("source", "?") for d in docs]
        return msg.content, retrieved, u.get("input_tokens", 0), u.get("output_tokens", 0), latency

    return run


def score_one(q, text, retrieved, judge_provider):
    """The capstone's metrics for a single (question, answer) pair."""
    out = {"retrieved": retrieved}
    if q["answerable"]:
        out["hit"] = R.path_matches_any(retrieved, q["expected_files"])
        n_c, n_r, n_m = R.score_citations(text, q["expected_files"], CORPUS)
        out["citations"] = {"total": n_c, "resolve": n_r, "match": n_m}
        out["judge_score"], out["judge_reason"] = R.judge(
            q["question"], q["keypoints"], text, judge_provider
        )
    else:
        out["declined"] = DECLINE_PHRASE in text
    return out


def aggregate(results, tokens_in, tokens_out):
    answerable = [r for r in results if "judge_score" in r]
    negatives = [r for r in results if "declined" in r]
    total_cites = sum(r["citations"]["total"] for r in answerable)
    totals = [i + o for i, o in zip(tokens_in, tokens_out)]
    return {
        "judged_correctness": round(sum(r["judge_score"] for r in answerable) / len(answerable), 3),
        "hit_at_k": round(sum(r["hit"] for r in answerable) / len(answerable), 3),
        "decline_accuracy": (round(sum(r["declined"] for r in negatives) / len(negatives), 3)
                             if negatives else None),
        "citation_resolve": round(sum(r["citations"]["resolve"] for r in answerable) / total_cites, 3) if total_cites else 0.0,
        "citation_match": round(sum(r["citations"]["match"] for r in answerable) / total_cites, 3) if total_cites else 0.0,
        "citations_per_answer": round(total_cites / len(answerable), 2),
        "tokens_out_p50": int(statistics.median(tokens_out)),
        "tokens_in_p50": int(statistics.median(tokens_in)),
        "tokens_total_p50": int(statistics.median(totals)),
        "n": len(results),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (smoke)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "parity.run.json"))
    args = ap.parse_args()

    if os.getenv("PROVIDER", "").lower() != "openai":
        raise SystemExit("Set PROVIDER=openai (both columns answer with gpt-4o-mini).")

    with open(GOLDEN, encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        gold = gold[: args.limit]

    answer_provider = get_provider("openai")            # gpt-4o-mini, askrepo answers
    judge_provider = get_provider("openai")             # gpt-4o-mini, grades both
    asklc_run = make_asklc_runner()

    rows = []
    ak_res, lc_res = [], []
    ak_in, ak_out, lc_in, lc_out = [], [], [], []
    for q in gold:
        ak_text, ak_ret, ak_i, ak_o, ak_lat = run_askrepo(q["question"], answer_provider)
        lc_text, lc_ret, lc_i, lc_o, lc_lat = asklc_run(q["question"])
        ak_score = score_one(q, ak_text, ak_ret, judge_provider)
        lc_score = score_one(q, lc_text, lc_ret, judge_provider)
        ak_res.append(ak_score); lc_res.append(lc_score)
        ak_in.append(ak_i); ak_out.append(ak_o); lc_in.append(lc_i); lc_out.append(lc_o)
        rows.append({
            "id": q["id"], "category": q["category"], "answerable": q["answerable"],
            "askrepo": {"answer": ak_text, "tokens": [ak_i, ak_o], **ak_score},
            "asklc": {"answer": lc_text, "tokens": [lc_i, lc_o], **lc_score},
        })
        mk = lambda s: (f"judge={s.get('judge_score')}" if "judge_score" in s
                        else f"declined={s.get('declined')}")
        print(f"  {q['id']:<8} askrepo {mk(ak_score):<14} | asklc {mk(lc_score):<14}", file=sys.stderr)

    ak_metrics = aggregate(ak_res, ak_in, ak_out)
    lc_metrics = aggregate(lc_res, lc_in, lc_out)

    print("\n=== parity (k=5, gpt-4o-mini, judge=gpt-4o-mini) ===")
    print(f"{'metric':<24}{'askrepo':>12}{'asklc':>12}{'delta':>12}")
    for key in ("judged_correctness", "hit_at_k", "decline_accuracy",
                "citation_resolve", "citation_match", "citations_per_answer",
                "tokens_out_p50", "tokens_in_p50", "tokens_total_p50"):
        a, b = ak_metrics[key], lc_metrics[key]
        d = f"{b - a:+.3f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"
        print(f"{key:<24}{str(a):>12}{str(b):>12}{d:>12}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"k": K, "n": len(gold), "askrepo": ak_metrics,
                   "asklc": lc_metrics, "questions": rows}, f, indent=1)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
