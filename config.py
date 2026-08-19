"""
config.py

Central configuration for the Liver RAG pipeline.

Place this file at the project ROOT (same level as rag/, models/, eval/,
data/, app.py). Every script imports its constants from here instead of
redefining its own paths/params, so changing one value here updates the
whole pipeline.
"""

import os
import pathlib


# ==========================================================================
# Environment
# ==========================================================================

# Set ENVIRONMENT=production on Railway.
# Locally it defaults to "development".
ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development",
)

IS_PRODUCTION = (
    ENVIRONMENT.lower() == "production"
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
    DATA_DIR
    / "s41392-024-02072-z.pdf"
)

CHUNKS_PATH = (
    DATA_DIR
    / "chunks.json"
)

EMBEDDINGS_PATH = (
    DATA_DIR
    / "embeddings.npy"
)

EMBEDDINGS_META_PATH = (
    DATA_DIR
    / "embeddings_meta.json"
)

BM25_INDEX_PATH = (
    DATA_DIR
    / "bm25_index.pkl"
)

GRAPH_TRIPLES_PATH = (
    DATA_DIR
    / "graph_triples.json"
)

QA_DATASET_PATH = (
    EVAL_DIR
    / "qa_dataset.json"
)

RESULTS_PATH = (
    EVAL_DIR
    / "results.json"
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

# Embedding model.
#
# Local:
#     BAAI/bge-large-en-v1.5
#
# Railway:
#     The query embedding is sent to Hugging Face when
#     USE_HF_INFERENCE_API=true.
#
# IMPORTANT:
# Changing this model requires rebuilding the corpus embeddings
# and Chroma vector database because embedding dimensions differ.

EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "BAAI/bge-large-en-v1.5",
)


# Local CrossEncoder model.
#
# This is ONLY used when:
#
#     USE_HF_INFERENCE_API=false
#
# On Railway we use Jina instead, so this model does not need
# to be loaded into Railway RAM.

RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


# ==========================================================================
# Hugging Face Inference API
# ==========================================================================

# When true:
#
#     Embedding:
#         Hugging Face API
#
#     Reranking:
#         Jina API
#
# This prevents torch + sentence-transformers from being loaded
# into the small Railway container.

USE_HF_INFERENCE_API = (
    os.getenv(
        "USE_HF_INFERENCE_API",
        "false",
    ).lower()
    == "true"
)


HF_API_TOKEN = os.getenv(
    "HF_API_TOKEN"
)


# ==========================================================================
# Jina Reranker
# ==========================================================================

# Jina is used for cross-encoder reranking when
# USE_HF_INFERENCE_API=true.

JINA_API_KEY = os.getenv(
    "JINA_API_KEY"
)


JINA_RERANKER_MODEL = os.getenv(
    "JINA_RERANKER_MODEL",
    "jina-reranker-v2-base-multilingual",
)


JINA_API_URL = (
    "https://api.jina.ai/v1/rerank"
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

# Keep a SINGLE worker on small Railway instances.
#
# Additional workers duplicate process memory.

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

# Number of candidates retrieved independently by:
#
#     Semantic search
#     BM25
#
CANDIDATE_K = 30


# Reciprocal Rank Fusion constant.

RRF_K = 60


# Number of candidates kept after RRF and before reranking.

RRF_TOP_K = 8


# Wider candidate pool for offline evaluation only.

EVAL_RRF_TOP_K = 20


# Final number of documents returned after reranking.

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

# Current Groq-compatible generation model.

GENERATION_MODEL_NAME = (
    "openai/gpt-oss-120b"
)


# Number of chunks used by answer_question().

RETRIEVAL_TOP_K = 5


# ==========================================================================
# Graph context
# ==========================================================================

# Keep graph context relatively small.
#
# A very large graph context can introduce weak/general facts such as:
#
#     INTRODUCTION
#     REFERENCES
#     unrelated entities
#
# which can result in noisy citations.

GRAPH_FACTS_LIMIT = 5


# ==========================================================================
# Confidence
# ==========================================================================

# Keep the existing threshold.
#
# IMPORTANT:
# If a result with a reranker score > 0.5 is still labeled
# "Low confidence", then the actual confidence calculation is
# probably happening elsewhere in generate.py.
#
# Do NOT blindly lower this value further.

LOW_CONFIDENCE_THRESHOLD = 0.5


# Minimum confidence for determining whether the question
# belongs to the source/document scope.

SCOPE_CONFIDENCE_FLOOR = 0.3


# ==========================================================================
# Generation retry configuration
# ==========================================================================

GENERATION_MAX_RETRIES = 3

GENERATION_RETRY_BACKOFF_S = 1.5
