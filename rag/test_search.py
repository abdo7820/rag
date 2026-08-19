"""Hybrid retrieval: Semantic (Chroma) + BM25 -> RRF fusion -> Cross-encoder rerank.

Run: python rag/test_search.py "What causes liver cirrhosis?"
"""
import math
import os
import pathlib
import pickle
import sys
import threading

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import chromadb
from chromadb.config import Settings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import (
    VECTOR_DB_DIR, COLLECTION_NAME, BM25_INDEX_PATH, EMBED_MODEL_NAME,
    RERANKER_MODEL_NAME, CANDIDATE_K, RRF_TOP_K, FINAL_TOP_K, RRF_K,
    USE_HF_INFERENCE_API, HF_API_TOKEN,
)
from bm25_index import simple_tokenize

# --------------------------------------------------------------------------
# Remote (Hugging Face Inference API) embedding + reranking.
#
# Used instead of loading SentenceTransformer/CrossEncoder + torch locally
# when USE_HF_INFERENCE_API=true (config.py). Same model weights, same
# scores — the only difference is where they run. This is what keeps a
# 512MB Railway instance from OOMing: no multi-hundred-MB model sits in
# this process's RAM at all.
# --------------------------------------------------------------------------
import requests

HF_API_URL = "https://api-inference.huggingface.co/models"


def _hf_headers() -> dict:
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is not set (.env) but USE_HF_INFERENCE_API=true.")
    return {"Authorization": f"Bearer {HF_API_TOKEN}"}


def _hf_embed_query(text: str) -> list[float]:
    """Feature-extraction call for a sentence-transformers model. HF's
    Inference API applies the model's own pooling for sentence-transformers
    repos, so this returns one already-pooled vector per input string."""
    resp = requests.post(
        f"{HF_API_URL}/{EMBED_MODEL_NAME}",
        headers=_hf_headers(),
        json={"inputs": [text], "options": {"wait_for_model": True}},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # Response is [[float, float, ...]] for a single input string.
    return data[0]


def _hf_rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Calls the cross-encoder as a text-pair classifier. Returns one
    relevance score per text, same ordering as the input list."""
    scores = []
    for text in texts:
        resp = requests.post(
            f"{HF_API_URL}/{RERANKER_MODEL_NAME}",
            headers=_hf_headers(),
            json={"inputs": {"text": query, "text_pair": text}, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        # text-classification pair output: [{"label": "...", "score": float}]
        score = result[0]["score"] if isinstance(result, list) else result.get("score", 0.0)
        scores.append(score)
    return scores


# --------------------------------------------------------------------------
# Cached, lazily-loaded singletons, guarded by locks.
#
# FastAPI serves sync routes from a threadpool, so several requests can hit
# an unloaded singleton at the same moment (most likely right after the
# server starts). Without a lock, each thread would start its own expensive
# model load — momentarily doubling/tripling RAM use, which is exactly the
# kind of spike that OOMs a small production instance. Double-checked
# locking here means the model loads exactly once no matter how many
# concurrent requests arrive first.
# --------------------------------------------------------------------------
_embed_model = None
_reranker_model = None
_chroma_collection = None
_bm25_payload = None

_embed_model_lock = threading.Lock()
_reranker_lock = threading.Lock()
_chroma_lock = threading.Lock()
_bm25_lock = threading.Lock()


def get_embed_model():
    """Only used when USE_HF_INFERENCE_API is false. Imports
    sentence-transformers lazily so it (and torch) are never even imported,
    let alone loaded, in the remote-inference path — that's what keeps RAM
    low enough for a free-tier Railway instance."""
    global _embed_model
    if _embed_model is None:
        with _embed_model_lock:
            if _embed_model is None:
                from sentence_transformers import SentenceTransformer
                _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_reranker():
    """Only used when USE_HF_INFERENCE_API is false."""
    global _reranker_model
    if _reranker_model is None:
        with _reranker_lock:
            if _reranker_model is None:
                from sentence_transformers import CrossEncoder
                print(f"Loading reranker: {RERANKER_MODEL_NAME}")
                _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_model


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        with _chroma_lock:
            if _chroma_collection is None:
                client = chromadb.PersistentClient(
                    path=str(VECTOR_DB_DIR),
                    settings=Settings(anonymized_telemetry=False),
                )
                try:
                    _chroma_collection = client.get_collection(COLLECTION_NAME)
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not load Chroma collection '{COLLECTION_NAME}'. Run rag/store.py first."
                    ) from exc
    return _chroma_collection


def get_bm25_payload() -> dict:
    global _bm25_payload
    if _bm25_payload is None:
        with _bm25_lock:
            if _bm25_payload is None:
                if not BM25_INDEX_PATH.exists():
                    raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}. Run rag/bm25_index.py first.")
                with open(BM25_INDEX_PATH, "rb") as f:
                    _bm25_payload = pickle.load(f)
    return _bm25_payload


def format_pages(meta: dict) -> str:
    start, end = meta.get("page_start"), meta.get("page_end")
    if start in (None, -1) and end in (None, -1):
        return "Unknown"
    if end in (None, -1) or start == end:
        return str(start)
    return f"{start}-{end}"


def print_reference(meta: dict):
    print("Reference:")
    print(f"  Source   : {meta.get('source', 'Unknown')}")
    print(f"  Section  : {meta.get('section', 'Unknown')}")
    print(f"  Pages    : {format_pages(meta)}")
    print(f"  DOI      : {meta.get('doi', 'Unknown')}")
    print(f"  Chunk ID : {meta.get('chunk_id', 'Unknown')}")


def print_preview(text: str, limit: int):
    print("\nText:")
    print(text[:limit].replace("\n", " ") + ("..." if len(text) > limit else ""))


def semantic_search(query: str):
    print("\n=== Semantic (Chroma) results ===")

    collection = get_chroma_collection()

    if USE_HF_INFERENCE_API:
        query_embedding = [_hf_embed_query(query)]
    else:
        model = get_embed_model()
        query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=CANDIDATE_K,
        include=["documents", "distances", "metadatas"],
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        print("No semantic results found.")
        return []

    retrieved = []
    for rank, (doc, dist, meta) in enumerate(zip(docs, distances, metadatas), start=1):
        meta = meta or {}
        retrieved.append({"chunk_id": meta.get("chunk_id"), "text": doc, "metadata": meta, "distance": dist, "rank": rank})
        if rank <= 10:
            print(f"\n[{rank}] distance={dist:.4f}")
            print_reference(meta)
            print_preview(doc, 400)

    return retrieved


def bm25_search(query: str):
    print("\n=== BM25 results ===")

    payload = get_bm25_payload()
    bm25, texts, metadata, chunk_ids = payload["bm25"], payload["texts"], payload.get("metadata", []), payload.get("chunk_ids", [])
    scores = bm25.get_scores(simple_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CANDIDATE_K]

    if not ranked:
        print("No BM25 results found.")
        return []

    retrieved = []
    for rank, idx in enumerate(ranked, start=1):
        reference = metadata[idx] if idx < len(metadata) else {
            "chunk_id": chunk_ids[idx] if idx < len(chunk_ids) else idx,
            "source": "Unknown", "doi": "Unknown", "section": "Unknown",
            "page_start": None, "page_end": None,
        }
        retrieved.append({
            "chunk_id": reference.get("chunk_id", idx),
            "text": texts[idx], "metadata": reference, "score": scores[idx], "rank": rank,
        })
        if rank <= 10:
            print(f"\n[{rank}] score={scores[idx]:.4f}")
            print_reference(reference)
            print_preview(texts[idx], 400)

    return retrieved


def reciprocal_rank_fusion(semantic_results, bm25_results):
    """score(d) = sum(1 / (RRF_K + rank))"""
    fused = {}

    def add(results, rank_key, extra_key, extra_field):
        for r in results:
            cid = r["chunk_id"]
            if cid is None:
                continue
            entry = fused.setdefault(cid, {
                "chunk_id": cid, "text": r["text"], "metadata": r["metadata"], "rrf_score": 0.0,
                "semantic_rank": None, "bm25_rank": None, "semantic_distance": None, "bm25_score": None,
            })
            entry["rrf_score"] += 1.0 / (RRF_K + r["rank"])
            entry[rank_key] = r["rank"]
            entry[extra_key] = r[extra_field]

    add(semantic_results, "semantic_rank", "semantic_distance", "distance")
    add(bm25_results, "bm25_rank", "bm25_score", "score")

    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:RRF_TOP_K]


def rerank_results(query: str, results: list[dict]):
    print("\n" + "=" * 70)
    print("=== Cross-Encoder Reranking ===")
    print("=" * 70)

    if not results:
        print("No results available for reranking.")
        return []

    if USE_HF_INFERENCE_API:
        texts = [r["text"] for r in results]
        # HF's text-classification pair endpoint already returns a
        # calibrated 0-1 probability, unlike the raw logits the local
        # CrossEncoder.predict() returns — no sigmoid squash needed here.
        scores = _hf_rerank_scores(query, texts)
    else:
        reranker = get_reranker()
        pairs = [[query, r["text"]] for r in results]
        raw_scores = reranker.predict(pairs, show_progress_bar=True, batch_size=8)
        # ms-marco-MiniLM-L-6-v2 outputs unbounded raw logits, not 0-1
        # probabilities. Squash with a sigmoid to keep downstream
        # confidence-threshold logic valid regardless of the reranker model.
        scores = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]

    reranked = [dict(r, reranker_score=float(s)) for r, s in zip(results, scores)]
    reranked.sort(key=lambda x: x["reranker_score"], reverse=True)
    final = reranked[:FINAL_TOP_K]

    for rank, r in enumerate(final, start=1):
        print(f"\n[{rank}] reranker_score={r['reranker_score']:.4f}")
        print(f"RRF score     : {r['rrf_score']:.6f}")
        print(f"Semantic rank : {r['semantic_rank']}")
        print(f"BM25 rank     : {r['bm25_rank']}")
        print_reference(r["metadata"])
        print_preview(r["text"], 700)

    return final


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What causes liver cirrhosis?"
    print(f"Query: {query}")

    semantic_results = semantic_search(query)
    bm25_results = bm25_search(query)
    rrf_results = reciprocal_rank_fusion(semantic_results, bm25_results)

    print("\n" + "=" * 70)
    print("=== RRF candidates ===")
    print("=" * 70)
    for rank, r in enumerate(rrf_results, start=1):
        print(f"\n[{rank}] RRF={r['rrf_score']:.6f} | chunk={r['chunk_id']}")
        print(f"Semantic rank: {r['semantic_rank']}")
        print(f"BM25 rank: {r['bm25_rank']}")
        print_reference(r["metadata"])

    final_results = rerank_results(query, rrf_results)

    print("\n" + "=" * 70)
    print("=== FINAL RETRIEVAL RESULTS ===")
    print("=" * 70)
    for rank, r in enumerate(final_results, start=1):
        print(f"\n[{rank}] score={r['reranker_score']:.4f}")
        print_reference(r["metadata"])
        print_preview(r["text"], 800)
