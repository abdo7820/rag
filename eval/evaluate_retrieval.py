"""
eval/evaluate_retrieval.py

Runs the hybrid retrieval pipeline (semantic + BM25 -> RRF fusion, optionally
+ cross-encoder reranking) against a ground-truth QA dataset and reports:

  - Hit Rate@K       (fraction of questions with a relevant chunk in the top K)
  - avg Precision@K  (fraction of the top K results that are actually relevant)
  - MRR              (mean reciprocal rank of the first relevant chunk)
  - Avg latency per query

Metrics are computed twice — once for RRF-only results and once after
reranking — so the two stages can be compared directly, and a failure list
with likely causes is printed for questions that scored zero in both.

NOTE: this script's candidate pool (EVAL_RRF_TOP_K, see config.py) is wider
than what production actually keeps (RRF_TOP_K) — it stress-tests more of
the ranking on purpose. Read these numbers as an upper bound on retrieval
quality, not a live measurement of exactly what /search returns.

Run:
    python eval/evaluate_retrieval.py                # full eval incl. reranking
    python eval/evaluate_retrieval.py --skip-rerank   # faster, RRF-only
    python eval/evaluate_retrieval.py --top-k 3
"""
import argparse
import json
import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "rag"))
sys.path.insert(0, str(BASE_DIR))

from bm25_index import simple_tokenize  # noqa: E402
from config import (  # noqa: E402
    QA_DATASET_PATH as QA_PATH, RESULTS_PATH, VECTOR_DB_DIR, COLLECTION_NAME,
    BM25_INDEX_PATH, EMBED_MODEL_NAME, RERANKER_MODEL_NAME, CANDIDATE_K,
    EVAL_RRF_TOP_K as RRF_TOP_K, RRF_K, KEYWORD_COVERAGE_THRESHOLD,
)


def load_qa_dataset() -> list[dict]:
    if not QA_PATH.exists():
        raise FileNotFoundError(f"Could not find {QA_PATH}.")
    return json.loads(QA_PATH.read_text(encoding="utf-8"))["questions"]


def is_relevant(text: str, expected_keywords: list[str]) -> bool:
    text_lower = text.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return (matched / len(expected_keywords)) >= KEYWORD_COVERAGE_THRESHOLD


def keyword_hit_fraction(text: str, expected_keywords: list[str]) -> float:
    """Same coverage math as is_relevant, but returns the fraction instead
    of a boolean — used for failure diagnostics (how close did we get?)."""
    text_lower = text.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return matched / len(expected_keywords) if expected_keywords else 0.0


def semantic_search(model, collection, query: str) -> list[dict]:
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding, n_results=CANDIDATE_K,
        include=["documents", "distances", "metadatas"],
    )
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return [
        {"chunk_id": (m or {}).get("chunk_id"), "text": doc, "rank": rank}
        for rank, (doc, m) in enumerate(zip(docs, metadatas), start=1)
    ]


def bm25_search(bm25_payload, query: str) -> list[dict]:
    bm25, texts, metadata = bm25_payload["bm25"], bm25_payload["texts"], bm25_payload.get("metadata", [])
    scores = bm25.get_scores(simple_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CANDIDATE_K]

    return [
        {"chunk_id": (metadata[idx] if idx < len(metadata) else {}).get("chunk_id", idx),
         "text": texts[idx], "rank": rank}
        for rank, idx in enumerate(ranked, start=1)
    ]


def reciprocal_rank_fusion(semantic_results, bm25_results) -> list[dict]:
    fused = {}

    def add(results, rank_key):
        for r in results:
            cid = r["chunk_id"]
            if cid is None:
                continue
            entry = fused.setdefault(cid, {"chunk_id": cid, "text": r["text"], "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (RRF_K + r["rank"])

    add(semantic_results, "semantic_rank")
    add(bm25_results, "bm25_rank")

    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:RRF_TOP_K]


def rerank(reranker, query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    reranked = [dict(c, reranker_score=float(s)) for c, s in zip(candidates, scores)]
    reranked.sort(key=lambda x: x["reranker_score"], reverse=True)
    return reranked


def score_ranked_list(ranked: list[dict], expected_keywords: list[str], top_k: int) -> tuple[bool, int | None]:
    """Returns (hit_in_top_k, rank_of_first_relevant_or_None)."""
    top = ranked[:top_k]
    for i, item in enumerate(top, start=1):
        if is_relevant(item["text"], expected_keywords):
            return True, i
    return False, None


def precision_at_k(ranked: list[dict], expected_keywords: list[str], top_k: int) -> float:
    top = ranked[:top_k]
    if not top:
        return 0.0
    relevant = sum(1 for item in top if is_relevant(item["text"], expected_keywords))
    return relevant / len(top)


def best_coverage_in_top_k(ranked: list[dict], expected_keywords: list[str], top_k: int) -> float:
    """Highest keyword-coverage fraction achieved by any chunk in the top K —
    used to distinguish "retrieval found something close but just under the
    threshold" from "retrieval found nothing related at all"."""
    top = ranked[:top_k]
    if not top:
        return 0.0
    return max((keyword_hit_fraction(item["text"], expected_keywords) for item in top), default=0.0)


def summarize(name: str, per_question: list[dict], top_k: int) -> dict:
    n = len(per_question)
    hits = sum(1 for q in per_question if q["hit"])
    reciprocal_ranks = [1.0 / q["rank"] if q["rank"] else 0.0 for q in per_question]
    avg_latency = sum(q["latency_s"] for q in per_question) / n if n else 0.0
    avg_precision = sum(q.get("precision", 0.0) for q in per_question) / n if n else 0.0

    return {
        "stage": name,
        "n_questions": n,
        f"hit_rate@{top_k}": round(hits / n, 3) if n else 0.0,
        f"avg_precision@{top_k}": round(avg_precision, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3) if n else 0.0,
        "avg_latency_s": round(avg_latency, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5, help="cutoff K for Hit Rate@K")
    parser.add_argument("--skip-rerank", action="store_true", help="evaluate RRF-only, skip cross-encoder")
    args = parser.parse_args()

    questions = load_qa_dataset()
    print(f"Loaded {len(questions)} questions from {QA_PATH}")
    print(f"RRF candidate pool (eval): top {RRF_TOP_K} — production uses a narrower pool, see config.py")

    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(f"Could not load Chroma collection '{COLLECTION_NAME}'. Run rag/store.py first.") from exc

    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}. Run rag/bm25_index.py first.")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_payload = pickle.load(f)

    reranker = None
    if not args.skip_rerank:
        print(f"Loading reranker: {RERANKER_MODEL_NAME} ...")
        reranker = CrossEncoder(RERANKER_MODEL_NAME)

    rrf_results_per_q = []
    reranked_results_per_q = []
    failures = []

    print("\nRunning evaluation ...")
    for q in questions:
        query, expected = q["question"], q["expected_keywords"]

        t0 = time.time()
        sem = semantic_search(embed_model, collection, query)
        bm = bm25_search(bm25_payload, query)
        fused = reciprocal_rank_fusion(sem, bm)
        rrf_latency = time.time() - t0

        hit, rank = score_ranked_list(fused, expected, args.top_k)
        precision = precision_at_k(fused, expected, args.top_k)
        rrf_results_per_q.append({"id": q["id"], "hit": hit, "rank": rank, "precision": precision, "latency_s": rrf_latency})

        rerank_hit, rerank_rank = None, None
        reranked = fused
        if reranker is not None:
            t1 = time.time()
            reranked = rerank(reranker, query, fused)
            rerank_latency = (time.time() - t1) + rrf_latency

            rerank_hit, rerank_rank = score_ranked_list(reranked, expected, args.top_k)
            rerank_precision = precision_at_k(reranked, expected, args.top_k)
            reranked_results_per_q.append({"id": q["id"], "hit": rerank_hit, "rank": rerank_rank, "precision": rerank_precision, "latency_s": rerank_latency})

        overall_hit = hit or (rerank_hit or False)
        if not overall_hit:
            best_cov = best_coverage_in_top_k(reranked, expected, args.top_k)
            failures.append({
                "id": q["id"], "question": query, "expected_keywords": expected,
                "best_keyword_coverage": round(best_cov, 2),
            })

        status = "OK" if overall_hit else "MISS"
        print(f"  [{status}] Q{q['id']}: {query}")

    print("\n" + "=" * 70)
    print("=== SUMMARY ===")
    print("=" * 70)

    rrf_summary = summarize("RRF only (no rerank)", rrf_results_per_q, args.top_k)
    print(json.dumps(rrf_summary, indent=2))

    rerank_summary = None
    if reranker is not None:
        rerank_summary = summarize("RRF + cross-encoder rerank", reranked_results_per_q, args.top_k)
        print(json.dumps(rerank_summary, indent=2))

        print("\n--- Trade-off: reranking vs. RRF-only ---")
        hr_delta = rerank_summary[f"hit_rate@{args.top_k}"] - rrf_summary[f"hit_rate@{args.top_k}"]
        mrr_delta = rerank_summary["mrr"] - rrf_summary["mrr"]
        latency_delta = rerank_summary["avg_latency_s"] - rrf_summary["avg_latency_s"]
        print(f"Hit Rate@{args.top_k} change: {hr_delta:+.3f}")
        print(f"MRR change: {mrr_delta:+.3f}")
        print(f"Extra latency per query: {latency_delta:+.3f}s")

    if failures:
        print(f"\n--- Failure analysis ({len(failures)}/{len(questions)} questions with no relevant hit) ---")
        for f in failures:
            cov = f["best_keyword_coverage"]
            if cov == 0.0:
                cause = "no retrieved chunk mentioned ANY expected keyword — likely a genuine retrieval miss (topic not well represented in the index, or query phrasing very different from the source text)."
            elif cov < KEYWORD_COVERAGE_THRESHOLD:
                cause = (f"closest chunk covered {cov:.0%} of expected keywords (below the "
                         f"{KEYWORD_COVERAGE_THRESHOLD:.0%} threshold) — likely a near-miss: the answer may be "
                         f"split across chunk boundaries, or the expected_keywords list is stricter than the "
                         f"source wording.")
            else:
                cause = "a relevant-looking chunk was found but ranked below top_k — a ranking issue, not a retrieval gap."
            print(f"  Q{f['id']}: {f['question']}")
            print(f"      expected keywords: {f['expected_keywords']}")
            print(f"      likely cause: {cause}")
    else:
        print("\nNo failures — every question found a relevant chunk.")

    report = {
        "top_k": args.top_k,
        "rrf_candidate_pool_eval": RRF_TOP_K,
        "keyword_coverage_threshold": KEYWORD_COVERAGE_THRESHOLD,
        "rrf_only": rrf_summary,
        "reranked": rerank_summary,
        "failures": failures,
        "per_question_rrf": rrf_results_per_q,
        "per_question_reranked": reranked_results_per_q,
    }
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
