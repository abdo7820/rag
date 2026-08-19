"""Hybrid retrieval: Semantic (Chroma) + BM25 -> RRF fusion -> Cross-encoder rerank.

Run:
    python rag/test_search.py "What causes liver cirrhosis?"
"""

import math
import os
import pathlib
import pickle
import sys
import threading

import chromadb
from chromadb.config import Settings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

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
import requests


# ==========================================================================
# Hugging Face Inference
# ==========================================================================
#
# Railway:
#   USE_HF_INFERENCE_API=true
#
# This avoids loading:
#   - torch
#   - sentence-transformers
#   - embedding model
#   - reranker model
#
# locally.
#
# We use the current Hugging Face Router instead of the deprecated:
#   api-inference.huggingface.co
#
# ==========================================================================

HF_API_URL = "https://router.huggingface.co/hf-inference/models"


def _hf_headers() -> dict:
    """Build clean Hugging Face authorization headers."""

    if not HF_API_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN is not set (.env) "
            "but USE_HF_INFERENCE_API=true."
        )

    token = HF_API_TOKEN.strip()

    if not token:
        raise RuntimeError(
            "HF_API_TOKEN is empty."
        )

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# ==========================================================================
# Hugging Face Embedding
# ==========================================================================

def _hf_embed_query(text: str) -> list[float]:
    """
    Generate an embedding using Hugging Face Inference.

    Uses the configured embedding model, for example:
        BAAI/bge-large-en-v1.5
    """

    url = (
        f"{HF_API_URL}/{EMBED_MODEL_NAME}"
        "/pipeline/feature-extraction"
    )

    response = requests.post(
        url,
        headers=_hf_headers(),
        json={
            "inputs": [text],
            "normalize": True,
        },
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            "Hugging Face embedding request failed "
            f"({response.status_code}): "
            f"{response.text[:1000]}"
        )

    data = response.json()

    # Expected:
    # [
    #     [0.01, 0.02, ...]
    # ]
    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], list)
        and data[0]
        and isinstance(data[0][0], (int, float))
    ):
        return [
            float(value)
            for value in data[0]
        ]

    # Some models/providers may return:
    #
    # [
    #     [
    #         [token_embedding...],
    #         [token_embedding...],
    #     ]
    # ]
    #
    # Fallback to first token vector.
    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], list)
        and data[0]
        and isinstance(data[0][0], list)
    ):
        return [
            float(value)
            for value in data[0][0]
        ]

    raise RuntimeError(
        "Unexpected Hugging Face embedding "
        f"response shape: {type(data).__name__}"
    )


# ==========================================================================
# Hugging Face Reranker
# ==========================================================================

def _hf_rerank_scores(
    query: str,
    texts: list[str],
) -> list[float]:
    """
    Run cross-encoder reranking through Hugging Face.

    Returns one relevance score for every text.
    """

    scores = []

    for text in texts:

        response = requests.post(
            f"{HF_API_URL}/{RERANKER_MODEL_NAME}",
            headers=_hf_headers(),
            json={
                "inputs": {
                    "text": query,
                    "text_pair": text,
                },
                "parameters": {
                    "function_to_apply": "sigmoid",
                },
            },
            timeout=60,
        )

        if not response.ok:
            raise RuntimeError(
                "Hugging Face reranker request failed "
                f"({response.status_code}): "
                f"{response.text[:1000]}"
            )

        result = response.json()

        if (
            isinstance(result, list)
            and result
        ):
            item = result[0]

            if isinstance(item, dict):
                score = item.get(
                    "score",
                    0.0,
                )
            else:
                score = float(item)

        elif isinstance(result, dict):

            score = float(
                result.get(
                    "score",
                    0.0,
                )
            )

        else:

            raise RuntimeError(
                "Unexpected Hugging Face "
                f"reranker response: {result!r}"
            )

        scores.append(
            float(score)
        )

    return scores


# ==========================================================================
# Lazy local models
# ==========================================================================

_embed_model = None
_reranker_model = None
_chroma_collection = None
_bm25_payload = None

_embed_model_lock = threading.Lock()
_reranker_lock = threading.Lock()
_chroma_lock = threading.Lock()
_bm25_lock = threading.Lock()


def get_embed_model():
    """
    Used only when USE_HF_INFERENCE_API=false.

    sentence-transformers is imported lazily so Railway does not need
    torch or sentence-transformers when using the HF API.
    """

    global _embed_model

    if _embed_model is None:

        with _embed_model_lock:

            if _embed_model is None:

                from sentence_transformers import (
                    SentenceTransformer
                )

                _embed_model = SentenceTransformer(
                    EMBED_MODEL_NAME
                )

    return _embed_model


def get_reranker():
    """
    Used only when USE_HF_INFERENCE_API=false.
    """

    global _reranker_model

    if _reranker_model is None:

        with _reranker_lock:

            if _reranker_model is None:

                from sentence_transformers import (
                    CrossEncoder
                )

                print(
                    f"Loading reranker: "
                    f"{RERANKER_MODEL_NAME}"
                )

                _reranker_model = CrossEncoder(
                    RERANKER_MODEL_NAME
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

                client = chromadb.PersistentClient(
                    path=str(VECTOR_DB_DIR),
                    settings=Settings(
                        anonymized_telemetry=False
                    ),
                )

                try:

                    _chroma_collection = (
                        client.get_collection(
                            COLLECTION_NAME
                        )
                    )

                except Exception as exc:

                    raise RuntimeError(
                        f"Could not load Chroma collection "
                        f"'{COLLECTION_NAME}'. "
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
                        f"BM25 index not found at "
                        f"{BM25_INDEX_PATH}. "
                        "Run rag/bm25_index.py first."
                    )

                with open(
                    BM25_INDEX_PATH,
                    "rb",
                ) as file:

                    _bm25_payload = pickle.load(
                        file
                    )

    return _bm25_payload


# ==========================================================================
# Metadata helpers
# ==========================================================================

def format_pages(meta: dict) -> str:

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


def print_reference(meta: dict):

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

    preview = text[:limit].replace(
        "\n",
        " ",
    )

    suffix = (
        "..."
        if len(text) > limit
        else ""
    )

    print(
        "\nText:"
    )

    print(
        preview + suffix
    )


# ==========================================================================
# Semantic Search
# ==========================================================================

def semantic_search(
    query: str,
):

    print(
        "\n=== Semantic (Chroma) results ==="
    )

    collection = get_chroma_collection()

    if USE_HF_INFERENCE_API:

        query_embedding = [
            _hf_embed_query(query)
        ]

    else:

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
        distance,
        metadata,
    ) in enumerate(
        zip(
            docs,
            distances,
            metadatas,
        ),
        start=1,
    ):

        metadata = metadata or {}

        retrieved.append(
            {
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "text": doc,
                "metadata": metadata,
                "distance": distance,
                "rank": rank,
            }
        )

        if rank <= 10:

            print(
                f"\n[{rank}] "
                f"distance={distance:.4f}"
            )

            print_reference(
                metadata
            )

            print_preview(
                doc,
                400,
            )

    return retrieved


# ==========================================================================
# BM25 Search
# ==========================================================================

def bm25_search(
    query: str,
):

    print(
        "\n=== BM25 results ==="
    )

    payload = get_bm25_payload()

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

        if idx < len(metadata):

            reference = metadata[idx]

        else:

            reference = {
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
# Reciprocal Rank Fusion
# ==========================================================================

def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
):
    """
    score(d) =
        sum(1 / (RRF_K + rank))
    """

    fused = {}

    def add(
        results,
        rank_key,
        extra_key,
        extra_field,
    ):

        for result in results:

            chunk_id = result[
                "chunk_id"
            ]

            if chunk_id is None:
                continue

            entry = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
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
                    + result["rank"]
                )
            )

            entry[
                rank_key
            ] = result["rank"]

            entry[
                extra_key
            ] = result[
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
        key=lambda x: x["rrf_score"],
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
        "\n" + "=" * 70
    )

    print(
        "=== Cross-Encoder Reranking ==="
    )

    print(
        "=" * 70
    )

    if not results:

        print(
            "No results available for reranking."
        )

        return []

    if USE_HF_INFERENCE_API:

        texts = [
            result["text"]
            for result in results
        ]

        scores = _hf_rerank_scores(
            query,
            texts,
        )

    else:

        reranker = get_reranker()

        pairs = [
            [
                query,
                result["text"],
            ]
            for result in results
        ]

        raw_scores = reranker.predict(
            pairs,
            show_progress_bar=True,
            batch_size=8,
        )

        scores = [
            1.0 / (
                1.0 + math.exp(-score)
            )
            for score in raw_scores
        ]

    reranked = [
        dict(
            result,
            reranker_score=float(score),
        )
        for result, score
        in zip(
            results,
            scores,
        )
    ]

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    final = reranked[
        :FINAL_TOP_K
    ]

    for rank, result in enumerate(
        final,
        start=1,
    ):

        print(
            f"\n[{rank}] "
            f"reranker_score="
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"RRF score     : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Semantic rank : "
            f"{result['semantic_rank']}"
        )

        print(
            f"BM25 rank     : "
            f"{result['bm25_rank']}"
        )

        print_reference(
            result["metadata"]
        )

        print_preview(
            result["text"],
            700,
        )

    return final


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    query = (
        " ".join(sys.argv[1:])
        or "What causes liver cirrhosis?"
    )

    print(
        f"Query: {query}"
    )

    semantic_results = semantic_search(
        query
    )

    bm25_results = bm25_search(
        query
    )

    rrf_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "=== RRF candidates ==="
    )

    print(
        "=" * 70
    )

    for rank, result in enumerate(
        rrf_results,
        start=1,
    ):

        print(
            f"\n[{rank}] "
            f"RRF={result['rrf_score']:.6f} "
            f"| chunk={result['chunk_id']}"
        )

        print(
            f"Semantic rank: "
            f"{result['semantic_rank']}"
        )

        print(
            f"BM25 rank: "
            f"{result['bm25_rank']}"
        )

        print_reference(
            result["metadata"]
        )

    final_results = rerank_results(
        query,
        rrf_results,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "=== FINAL RETRIEVAL RESULTS ==="
    )

    print(
        "=" * 70
    )

    for rank, result in enumerate(
        final_results,
        start=1,
    ):

        print(
            f"\n[{rank}] "
            f"score="
            f"{result['reranker_score']:.4f}"
        )

        print_reference(
            result["metadata"]
        )

        print_preview(
            result["text"],
            800,
        )
