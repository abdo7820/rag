"""
config.py

Central configuration for the Liver RAG pipeline.

This file is shared by:
    - rag/
    - models/
    - eval/
    - app.py

Production target:
    GitHub -> Railway

Railway is designed for low-RAM deployment:
    - Query embeddings -> Hugging Face Inference API
    - Reranking -> Jina API
    - No local BGE / CrossEncoder / torch loading
"""

import os
import pathlib


# ==========================================================================
# Environment
# ==========================================================================

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development",
).strip().lower()

IS_PRODUCTION = (
    ENVIRONMENT == "production"
)


# ==========================================================================
# Project layout
# ==========================================================================

BASE_DIR = pathlib.Path(
    __file__
).resolve().parent

DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "eval"
VECTOR_DB_DIR = BASE_DIR / "vectorDB"

MD_PATH = BASE_DIR / "liver_diseases.md"

PDF_PATH = (
    DATA_DIR /
    "s41392-024-02072-z.pdf"
)

CHUNKS_PATH = (
    DATA_DIR /
    "chunks.json"
)

EMBEDDINGS_PATH = (
    DATA_DIR /
    "embeddings.npy"
)

EMBEDDINGS_META_PATH = (
    DATA_DIR /
    "embeddings_meta.json"
)

BM25_INDEX_PATH = (
    DATA_DIR /
    "bm25_index.pkl"
)

GRAPH_TRIPLES_PATH = (
    DATA_DIR /
    "graph_triples.json"
)

QA_DATASET_PATH = (
    EVAL_DIR /
    "qa_dataset.json"
)

RESULTS_PATH = (
    EVAL_DIR /
    "results.json"
)


# ==========================================================================
# Source document metadata
# ==========================================================================

SOURCE_NAME = (
    "Liver diseases: epidemiology, causes, "
    "trends and predictions"
)

DOI = (
    "10.1038/s41392-024-02072-z"
)


# ==========================================================================
# Models
# ==========================================================================

# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------
#
# Corpus embeddings were generated using this model.
#
# IMPORTANT:
# If this value changes, the corpus embeddings and Chroma database
# must be rebuilt.
#
# Local:
#     SentenceTransformer loads this model.
#
# Railway:
#     Query embeddings are generated through Hugging Face API.
#

EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "BAAI/bge-large-en-v1.5",
).strip()


# --------------------------------------------------------------------------
# Local CrossEncoder
# --------------------------------------------------------------------------
#
# Used ONLY when:
#
#     USE_JINA_RERANKER_API=false
#
# Railway should NOT load this model.
#

RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
).strip()


# ==========================================================================
# Hugging Face Inference API
# ==========================================================================

# Railway:
#
#     USE_HF_INFERENCE_API=true
#
# This means:
#
#     Query
#       ↓
#     Hugging Face embedding API
#       ↓
#     Chroma
#
# The local BGE model is NOT loaded into RAM.

USE_HF_INFERENCE_API = (
    os.getenv(
        "USE_HF_INFERENCE_API",
        "false",
    ).strip().lower()
    == "true"
)


HF_API_TOKEN = os.getenv(
    "HF_API_TOKEN"
)

if HF_API_TOKEN:
    HF_API_TOKEN = HF_API_TOKEN.strip()


HF_API_BASE_URL = (
    "https://api-inference.huggingface.co/models"
)


# ==========================================================================
# Jina Reranker API
# ==========================================================================

# Production:
#
#     USE_JINA_RERANKER_API=true
#
# The reranker is executed remotely through Jina.
#
# This prevents:
#
#     torch
#     CrossEncoder
#     reranker weights
#
# from being loaded into Railway RAM.

USE_JINA_RERANKER_API = (
    os.getenv(
        "USE_JINA_RERANKER_API",
        "false",
    ).strip().lower()
    == "true"
)


JINA_API_KEY = os.getenv(
    "JINA_API_KEY"
)

if JINA_API_KEY:
    JINA_API_KEY = JINA_API_KEY.strip()


JINA_RERANKER_MODEL = os.getenv(
    "JINA_RERANKER_MODEL",
    "jina-reranker-v2-base-multilingual",
).strip()


JINA_API_URL = (
    "https://api.jina.ai/v1/rerank"
)


# --------------------------------------------------------------------------
# Remote API retry / timeout configuration
# --------------------------------------------------------------------------

REMOTE_API_TIMEOUT_SECONDS = int(
    os.getenv(
        "REMOTE_API_TIMEOUT_SECONDS",
        "60",
    )
)


REMOTE_API_MAX_RETRIES = int(
    os.getenv(
        "REMOTE_API_MAX_RETRIES",
        "2",
    )
)


# ==========================================================================
# Production model configuration validation
# ==========================================================================

# These checks intentionally happen only when production starts.
#
# They prevent Railway from silently starting with missing API keys.

if IS_PRODUCTION:

    if USE_HF_INFERENCE_API and not HF_API_TOKEN:

        raise RuntimeError(
            "Production configuration error: "
            "USE_HF_INFERENCE_API=true but "
            "HF_API_TOKEN is missing."
        )

    if (
        USE_JINA_RERANKER_API
        and not JINA_API_KEY
    ):

        raise RuntimeError(
            "Production configuration error: "
            "USE_JINA_RERANKER_API=true but "
            "JINA_API_KEY is missing."
        )


# ==========================================================================
# Graph extraction
# ==========================================================================

GRAPH_EXTRACTION_MODEL_NAME = (
    "llama-3.1-8b-instant"
)


# ==========================================================================
# Processing
# ==========================================================================

EMBED_BATCH_SIZE = 16

CHROMA_INSERT_BATCH = 100


# ==========================================================================
# Web server
# ==========================================================================

# Railway low-RAM deployment:
#
# Keep exactly ONE worker.
#
# Multiple workers duplicate:
#     FastAPI
#     Chroma
#     BM25
#     Python process memory
#
# and can cause OOM.

WEB_CONCURRENCY = int(
    os.getenv(
        "WEB_CONCURRENCY",
        "1",
    )
)


# ==========================================================================
# Vector store
# ==========================================================================

COLLECTION_NAME = (
    "liver_diseases"
)


# ==========================================================================
# Retrieval parameters
# ==========================================================================

# Candidates retrieved independently by:
#
#     Semantic Search
#     BM25
#

CANDIDATE_K = 30


# Reciprocal Rank Fusion constant.

RRF_K = 60


# Production:
#
# Candidates kept after RRF and sent to reranker.

RRF_TOP_K = 8


# Evaluation only.

EVAL_RRF_TOP_K = 20


# Final results after reranking.

FINAL_TOP_K = 5


# Evaluation keyword coverage threshold.

KEYWORD_COVERAGE_THRESHOLD = 0.6


# ==========================================================================
# Chunking
# ==========================================================================

CHUNK_SIZE = 512

CHUNK_OVERLAP = 80


# ==========================================================================
# Generation
# ==========================================================================

GENERATION_MODEL_NAME = (
    "openai/gpt-oss-120b"
)


# Number of retrieved chunks used by /ask.

RETRIEVAL_TOP_K = 5


# ==========================================================================
# Graph context
# ==========================================================================

GRAPH_FACTS_LIMIT = 5


# ==========================================================================
# Confidence
# ==========================================================================

LOW_CONFIDENCE_THRESHOLD = 0.5

SCOPE_CONFIDENCE_FLOOR = 0.3


# ==========================================================================
# Generation retry configuration
# ==========================================================================

GENERATION_MAX_RETRIES = 3

GENERATION_RETRY_BACKOFF_S = 1.5
