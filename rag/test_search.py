"""
Hybrid retrieval:
Semantic (Chroma) + BM25 -> RRF fusion -> Reranking.

Reranking modes:

1. Jina API:
   USE_JINA_RERANKER_API=true

   Uses:
       jina-reranker-v2-base-multilingual

   Recommended for Railway / low-RAM deployment.

2. Hugging Face Inference API:
   USE_HF_INFERENCE_API=true

   Used for query embeddings.

3. Local:
   USE_JINA_RERANKER_API=false
   USE_HF_INFERENCE_API=false

   Loads SentenceTransformer + CrossEncoder locally.

Run:
    python rag/test_search.py "What causes liver cirrhosis?"
"""

import math
import os
import pathlib
import pickle
import sys
import threading
import time

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# --------------------------------------------------------------------------
# Third-party imports
# --------------------------------------------------------------------------

import chromadb
import requests

from chromadb.config import Settings


# --------------------------------------------------------------------------
# Project path
# --------------------------------------------------------------------------

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parent.parent),
)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

from config import (
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    BM25_INDEX_PATH,
    EMBED_MODEL_NAME,
    RERANKER_MODEL_NAME,
    CANDIDATE_K,
    RRF_TOP_K,
    FINAL_TOP_K,
    RRF_K,
    USE_HF_INFERENCE_API,
    HF_API_TOKEN,
)

from bm25_index import simple_tokenize


# ==========================================================================
# Remote embedding configuration
# ==========================================================================

HF_API_URL = "https://api-inference.huggingface.co/models"


def _clean_env(value):
    """
    Removes accidental spaces/newlines from environment variables.

    This is important because a token copied into Railway or .env
    may accidentally contain whitespace.
    """

    if value is None:
        return None

    return value.strip()


# ==========================================================================
# Jina configuration
# ==========================================================================

USE_JINA_RERANKER_API = (
    os.getenv(
        "USE_JINA_RERANKER_API",
        "false",
    ).strip().lower()
    == "true"
)

JINA_API_KEY = _clean_env(
    os.getenv("JINA_API_KEY")
)

JINA_RERANKER_MODEL = os.getenv(
    "JINA_RERANKER_MODEL",
    "jina-reranker-v2-base-multilingual",
).strip()

JINA_RERANK_URL = (
    "https://api.jina.ai/v1/rerank"
)


# ==========================================================================
# HTTP settings
# ==========================================================================

REMOTE_TIMEOUT_SECONDS = int(
    os.getenv(
        "REMOTE_API_TIMEOUT_SECONDS",
        "60",
    )
)

REMOTE_MAX_RETRIES = int(
    os.getenv(
        "REMOTE_API_MAX_RETRIES",
        "2",
    )
)


# ==========================================================================
# Hugging Face helpers
# ==========================================================================

def _hf_headers() -> dict:
    token = _clean_env(HF_API_TOKEN)

    if not token:
        raise RuntimeError(
            "HF_API_TOKEN is not set while "
            "USE_HF_INFERENCE_API=true."
        )

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# ==========================================================================
# HF embedding
# ==========================================================================

def _hf_embed_query(
    text: str,
) -> list[float]:
    """
    Generate one query embedding using Hugging Face.

    This is used only for query-time embedding.

    The corpus embeddings were generated beforehand and
    stored in Chroma.
    """

    last_error = None

    for attempt in range(
        1,
        REMOTE_MAX_RETRIES + 1,
    ):

        try:

            resp = requests.post(
                f"{HF_API_URL}/{EMBED_MODEL_NAME}",
                headers=_hf_headers(),
                json={
                    "inputs": [text],
                    "options": {
                        "wait_for_model": True
                    },
                },
                timeout=REMOTE_TIMEOUT_SECONDS,
            )

            if not resp.ok:

                raise RuntimeError(
                    "Hugging Face embedding request "
                    f"failed ({resp.status_code}): "
                    f"{resp.text[:1000]}"
                )

            data = resp.json()

            # Expected:
            # [[float, float, ...]]
            if (
                not isinstance(data, list)
                or not data
                or not isinstance(data[0], list)
            ):

                raise RuntimeError(
                    "Unexpected Hugging Face embedding "
                    f"response: {data}"
                )

            return data[0]

        except Exception as exc:

            last_error = exc

            print(
                f"[warn] HF embedding attempt "
                f"{attempt}/{REMOTE_MAX_RETRIES} "
                f"failed: {exc}"
            )

            if attempt < REMOTE_MAX_RETRIES:

                time.sleep(
                    1.5 * attempt
                )

    raise RuntimeError(
        "Hugging Face embedding failed after "
        f"{REMOTE_MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ==========================================================================
# Jina headers
# ==========================================================================

def _jina_headers() -> dict:

    token = _clean_env(
        JINA_API_KEY
    )

    if not token:

        raise RuntimeError(
            "JINA_API_KEY is not set while "
            "USE_JINA_RERANKER_API=true."
        )

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ==========================================================================
# Jina reranker
# ==========================================================================

def _jina_rerank_scores(
    query: str,
    texts: list[str],
) -> list[float]:
    """
    Rerank all documents in ONE Jina API request.

    Jina API returns results containing:

        index
        relevance_score

    We map the returned scores back to the original
    document order.

    This avoids making one HTTP request per chunk.
    """

    if not texts:

        return []

    payload = {
        "model": JINA_RERANKER_MODEL,
        "query": query,
        "documents": texts,
        "top_n": len(texts),
        "return_documents": False,
    }

    last_error = None

    for attempt in range(
        1,
        REMOTE_MAX_RETRIES + 1,
    ):

        try:

            response = requests.post(
                JINA_RERANK_URL,
                headers=_jina_headers(),
                json=payload,
                timeout=REMOTE_TIMEOUT_SECONDS,
            )

            if not response.ok:

                raise RuntimeError(
                    "Jina reranker request failed "
                    f"({response.status_code}): "
                    f"{response.text[:1500]}"
                )

            data = response.json()

            results = data.get(
                "results",
                [],
            )

            if not results:

                raise RuntimeError(
                    "Jina reranker returned no results."
                )

            # Initialize with zeroes.
            scores = [0.0] * len(texts)

            for item in results:

                index = item.get("index")

                score = item.get(
                    "relevance_score"
                )

                if score is None:

                    # Some API responses may use score.
                    score = item.get(
                        "score",
                        0.0,
                    )

                if index is None:
                    continue

                index = int(index)

                if (
                    0 <= index < len(scores)
                ):

                    scores[index] = float(
                        score
                    )

            return scores

        except Exception as exc:

            last_error = exc

            print(
                f"[warn] Jina reranker attempt "
                f"{attempt}/{REMOTE_MAX_RETRIES} "
                f"failed: {exc}"
            )

            if attempt < REMOTE_MAX_RETRIES:

                time.sleep(
                    1.5 * attempt
                )

    raise RuntimeError(
        "Jina reranker failed after "
        f"{REMOTE_MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ==========================================================================
# Local model singletons
# ==========================================================================

_embed_model = None
_reranker_model = None
_chroma_collection = None
_bm25_payload = None


_embed_model_lock = threading.Lock()
_reranker_lock = threading.Lock()
_chroma_lock = threading.Lock()
_bm25_lock = threading.Lock()


# ==========================================================================
# Local embedding model
# ==========================================================================

def get_embed_model():

    global _embed_model

    if USE_HF_INFERENCE_API:

        raise RuntimeError(
            "get_embed_model() should not be called "
            "when USE_HF_INFERENCE_API=true."
        )

    if _embed_model is None:

        with _embed_model_lock:

            if _embed_model is None:

                from sentence_transformers import (
                    SentenceTransformer
                )

                print(
                    f"Loading embedding model: "
                    f"{EMBED_MODEL_NAME}"
                )

                _embed_model = (
                    SentenceTransformer(
                        EMBED_MODEL_NAME
                    )
                )

    return _embed_model


# ==========================================================================
# Local CrossEncoder
# ==========================================================================

def get_reranker():

    global _reranker_model

    if USE_JINA_RERANKER_API:

        raise RuntimeError(
            "get_reranker() should not be called "
            "when USE_JINA_RERANKER_API=true."
        )

    if _reranker_model is None:

        with _reranker_lock:

            if _reranker_model is None:

                from sentence_transformers import (
                    CrossEncoder
                )

                print(
                    f"Loading local reranker: "
                    f"{RERANKER_MODEL_NAME}"
                )

                _reranker_model = (
                    CrossEncoder(
                        RERANKER_MODEL_NAME
                    )
                )

    return _reranker_model


# ==========================================================================
# Chroma
# ==========================================================================

def get_chroma_collection():

    global _chroma_collection

    if _chroma_collection is None:

        with _chroma_lock:

            if _chroma_collection is None:

                client = (
                    chromadb.PersistentClient(
                        path=str(
                            VECTOR_DB_DIR
                        ),
                        settings=Settings(
                            anonymized_telemetry=False
                        ),
                    )
                )

                try:

                    _chroma_collection = (
                        client.get_collection(
                            COLLECTION_NAME
                        )
                    )

                except Exception as exc:

                    raise RuntimeError(
                        "Could not load Chroma "
                        f"collection '{COLLECTION_NAME}'. "
                        "Run rag/store.py first."
                    ) from exc

    return _chroma_collection


# ==========================================================================
# BM25
# ==========================================================================

def get_bm25_payload() -> dict:

    global _bm25_payload

    if _bm25_payload is None:

        with _bm25_lock:

            if _bm25_payload is None:

                if not BM25_INDEX_PATH.exists():

                    raise FileNotFoundError(
                        "BM25 index not found at "
                        f"{BM25_INDEX_PATH}. "
                        "Run rag/bm25_index.py first."
                    )

                with open(
                    BM25_INDEX_PATH,
                    "rb",
                ) as f:

                    _bm25_payload = (
                        pickle.load(f)
                    )

    return _bm25_payload


# ==========================================================================
# Formatting
# ==========================================================================

def format_pages(
    meta: dict,
) -> str:

    start = meta.get(
        "page_start"
    )

    end = meta.get(
        "page_end"
    )

    if (
        start in (None, -1)
        and end in (None, -1)
    ):

        return "Unknown"

    if (
        end in (None, -1)
        or start == end
    ):

        return str(start)

    return f"{start}-{end}"


def print_reference(
    meta: dict,
):

    print("Reference:")

    print(
        f"  Source   : "
        f"{meta.get('source', 'Unknown')}"
    )

    print(
        f"  Section  : "
        f"{meta.get('section', 'Unknown')}"
    )

    print(
        f"  Pages    : "
        f"{format_pages(meta)}"
    )

    print(
        f"  DOI      : "
        f"{meta.get('doi', 'Unknown')}"
    )

    print(
        f"  Chunk ID : "
        f"{meta.get('chunk_id', 'Unknown')}"
    )


def print_preview(
    text: str,
    limit: int,
):

    clean = text[:limit].replace(
        "\n",
        " ",
    )

    print(
        "\nText:"
    )

    print(
        clean
        + (
            "..."
            if len(text) > limit
            else ""
        )
    )


# ==========================================================================
# Semantic search
# ==========================================================================

def semantic_search(
    query: str,
):

    print(
        "\n=== Semantic (Chroma) results ==="
    )

    collection = (
        get_chroma_collection()
    )

    if USE_HF_INFERENCE_API:

        print(
            "Embedding mode: "
            "Hugging Face Inference API"
        )

        query_embedding = [
            _hf_embed_query(query)
        ]

    else:

        print(
            "Embedding mode: LOCAL"
        )

        model = get_embed_model()

        query_embedding = (
            model.encode(
                [query],
                normalize_embeddings=True,
            ).tolist()
        )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=CANDIDATE_K,
        include=[
            "documents",
            "distances",
            "metadatas",
        ],
    )

    docs = results.get(
        "documents",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    if not docs:

        print(
            "No semantic results found."
        )

        return []

    retrieved = []

    for rank, (
        doc,
        dist,
        meta,
    ) in enumerate(
        zip(
            docs,
            distances,
            metadatas,
        ),
        start=1,
    ):

        meta = meta or {}

        retrieved.append(
            {
                "chunk_id": meta.get(
                    "chunk_id"
                ),
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "rank": rank,
            }
        )

        if rank <= 10:

            print(
                f"\n[{rank}] "
                f"distance={dist:.4f}"
            )

            print_reference(
                meta
            )

            print_preview(
                doc,
                400,
            )

    return retrieved


# ==========================================================================
# BM25 search
# ==========================================================================

def bm25_search(
    query: str,
):

    print(
        "\n=== BM25 results ==="
    )

    payload = (
        get_bm25_payload()
    )

    bm25 = payload["bm25"]
    texts = payload["texts"]

    metadata = payload.get(
        "metadata",
        [],
    )

    chunk_ids = payload.get(
        "chunk_ids",
        [],
    )

    scores = bm25.get_scores(
        simple_tokenize(query)
    )

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:CANDIDATE_K]

    if not ranked:

        print(
            "No BM25 results found."
        )

        return []

    retrieved = []

    for rank, idx in enumerate(
        ranked,
        start=1,
    ):

        reference = (
            metadata[idx]
            if idx < len(metadata)
            else {
                "chunk_id": (
                    chunk_ids[idx]
                    if idx < len(chunk_ids)
                    else idx
                ),
                "source": "Unknown",
                "doi": "Unknown",
                "section": "Unknown",
                "page_start": None,
                "page_end": None,
            }
        )

        retrieved.append(
            {
                "chunk_id": reference.get(
                    "chunk_id",
                    idx,
                ),
                "text": texts[idx],
                "metadata": reference,
                "score": scores[idx],
                "rank": rank,
            }
        )

        if rank <= 10:

            print(
                f"\n[{rank}] "
                f"score={scores[idx]:.4f}"
            )

            print_reference(
                reference
            )

            print_preview(
                texts[idx],
                400,
            )

    return retrieved


# ==========================================================================
# RRF
# ==========================================================================

def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
):

    """
    score(d) =
        sum(
            1 / (RRF_K + rank)
        )
    """

    fused = {}

    def add(
        results,
        rank_key,
        extra_key,
        extra_field,
    ):

        for r in results:

            cid = r[
                "chunk_id"
            ]

            if cid is None:
                continue

            entry = fused.setdefault(
                cid,
                {
                    "chunk_id": cid,
                    "text": r["text"],
                    "metadata": r[
                        "metadata"
                    ],
                    "rrf_score": 0.0,
                    "semantic_rank": None,
                    "bm25_rank": None,
                    "semantic_distance": None,
                    "bm25_score": None,
                },
            )

            entry[
                "rrf_score"
            ] += (
                1.0
                / (
                    RRF_K
                    + r["rank"]
                )
            )

            entry[
                rank_key
            ] = r["rank"]

            entry[
                extra_key
            ] = r[
                extra_field
            ]

    add(
        semantic_results,
        "semantic_rank",
        "semantic_distance",
        "distance",
    )

    add(
        bm25_results,
        "bm25_rank",
        "bm25_score",
        "score",
    )

    return sorted(
        fused.values(),
        key=lambda x: x[
            "rrf_score"
        ],
        reverse=True,
    )[:RRF_TOP_K]


# ==========================================================================
# Reranking
# ==========================================================================

def rerank_results(
    query: str,
    results: list[dict],
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "=== Reranking ==="
    )

    print(
        "=" * 70
    )

    if not results:

        print(
            "No results available "
            "for reranking."
        )

        return []

    # ----------------------------------------------------------------------
    # JINA API
    # ----------------------------------------------------------------------

    if USE_JINA_RERANKER_API:

        print(
            "Reranker mode: "
            "Jina API"
        )

        print(
            f"Model: "
            f"{JINA_RERANKER_MODEL}"
        )

        texts = [
            r["text"]
            for r in results
        ]

        scores = (
            _jina_rerank_scores(
                query,
                texts,
            )
        )

    # ----------------------------------------------------------------------
    # Hugging Face CrossEncoder
    # ----------------------------------------------------------------------

    else:

        print(
            "Reranker mode: "
            "LOCAL CrossEncoder"
        )

        reranker = (
            get_reranker()
        )

        pairs = [
            [
                query,
                r["text"],
            ]
            for r in results
        ]

        raw_scores = (
            reranker.predict(
                pairs,
                show_progress_bar=True,
                batch_size=8,
            )
        )

        scores = []

        for s in raw_scores:

            try:

                value = float(s)

                # Prevent overflow in exp()
                value = max(
                    -50.0,
                    min(
                        50.0,
                        value,
                    ),
                )

                score = (
                    1.0
                    / (
                        1.0
                        + math.exp(-value)
                    )
                )

            except Exception:

                score = 0.0

            scores.append(
                score
            )

    # ----------------------------------------------------------------------
    # Attach scores
    # ----------------------------------------------------------------------

    reranked = []

    for r, score in zip(
        results,
        scores,
    ):

        reranked.append(
            dict(
                r,
                reranker_score=float(
                    score
                ),
            )
        )

    reranked.sort(
        key=lambda x: x[
            "reranker_score"
        ],
        reverse=True,
    )

    final = reranked[
        :FINAL_TOP_K
    ]

    # ----------------------------------------------------------------------
    # Print final ranking
    # ----------------------------------------------------------------------

    for rank, r in enumerate(
        final,
        start=1,
    ):

        print(
            f"\n[{rank}] "
            f"reranker_score="
            f"{r['reranker_score']:.4f}"
        )

        print(
            f"RRF score     : "
            f"{r['rrf_score']:.6f}"
        )

        print(
            f"Semantic rank : "
            f"{r['semantic_rank']}"
        )

        print(
            f"BM25 rank     : "
            f"{r['bm25_rank']}"
        )

        print_reference(
            r["metadata"]
        )

        print_preview(
            r["text"],
            700,
        )

    return final


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    query = (
        " ".join(
            sys.argv[1:]
        )
        or
        "What causes liver cirrhosis?"
    )

    print(
        f"Query: {query}"
    )

    print(
        "\n=== CONFIGURATION ==="
    )

    print(
        "HF embedding API:",
        USE_HF_INFERENCE_API,
    )

    print(
        "Jina reranker API:",
        USE_JINA_RERANKER_API,
    )

    if USE_JINA_RERANKER_API:

        print(
            "Jina model:",
            JINA_RERANKER_MODEL,
        )

    print(
        "Embedding model:",
        EMBED_MODEL_NAME,
    )

    print(
        "Local reranker model:",
        RERANKER_MODEL_NAME,
    )

    # ----------------------------------------------------------------------
    # 1. Semantic
    # ----------------------------------------------------------------------

    semantic_results = (
        semantic_search(
            query
        )
    )

    # ----------------------------------------------------------------------
    # 2. BM25
    # ----------------------------------------------------------------------

    bm25_results = (
        bm25_search(
            query
        )
    )

    # ----------------------------------------------------------------------
    # 3. RRF
    # ----------------------------------------------------------------------

    rrf_results = (
        reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "=== RRF candidates ==="
    )

    print(
        "=" * 70
    )

    for rank, r in enumerate(
        rrf_results,
        start=1,
    ):

        print(
            f"\n[{rank}] "
            f"RRF={r['rrf_score']:.6f} "
            f"| chunk={r['chunk_id']}"
        )

        print(
            f"Semantic rank: "
            f"{r['semantic_rank']}"
        )

        print(
            f"BM25 rank: "
            f"{r['bm25_rank']}"
        )

        print_reference(
            r["metadata"]
        )

    # ----------------------------------------------------------------------
    # 4. Rerank
    # ----------------------------------------------------------------------

    final_results = (
        rerank_results(
            query,
            rrf_results,
        )
    )

    # ----------------------------------------------------------------------
    # 5. Final output
    # ----------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "=== FINAL RETRIEVAL RESULTS ==="
    )

    print(
        "=" * 70
    )

    for rank, r in enumerate(
        final_results,
        start=1,
    ):

        print(
            f"\n[{rank}] "
            f"score="
            f"{r['reranker_score']:.4f}"
        )

        print_reference(
            r["metadata"]
        )

        print_preview(
            r["text"],
            800,
        )
