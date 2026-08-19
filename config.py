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

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
# Set ENVIRONMENT=production on Railway (Settings -> Variables). Locally it
# defaults to "development" so nothing changes on your machine.
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# --------------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "eval"
VECTOR_DB_DIR = BASE_DIR / "vectorDB"

MD_PATH = BASE_DIR / "liver_diseases.md"
PDF_PATH = DATA_DIR / "s41392-024-02072-z.pdf"

CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_META_PATH = DATA_DIR / "embeddings_meta.json"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
GRAPH_TRIPLES_PATH = DATA_DIR / "graph_triples.json"

QA_DATASET_PATH = EVAL_DIR / "qa_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.json"

# --------------------------------------------------------------------------
# Source document metadata
# --------------------------------------------------------------------------
SOURCE_NAME = "Liver diseases: epidemiology, causes, trends and predictions"
DOI = "10.1038/s41392-024-02072-z"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# bge-large (~1.3GB in RAM) is the best-quality embedder, but combined with
# the reranker + torch it's the single biggest reason a small Railway
# instance OOMs. Override via env var without touching code: set
# EMBED_MODEL_NAME=BAAI/bge-base-en-v1.5 (~440MB) or bge-small-en-v1.5
# (~130MB) on Railway if you don't want to bump the RAM plan.
# IMPORTANT: changing this requires re-running embed.py + store.py — the
# vector dimensions change between bge-large/base/small, so the old
# vectorDB/ is incompatible and must be rebuilt from scratch.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# When true, rag/test_search.py calls the Hugging Face Inference API for
# query embedding + reranking at request time instead of loading
# SentenceTransformer/CrossEncoder (+ torch) into local RAM. Same model
# weights, same results — just runs on HF's servers instead of Railway's.
# This is what makes the free/512MB Railway tier viable: local RAM use
# drops to ~100-150MB (FastAPI + chromadb + the BM25 pickle in memory),
# since torch and both models are no longer loaded in-process at all.
# rag/embed.py (the one-time corpus-embedding step) is unaffected — run
# that locally on your own machine before deploying; only the per-query
# encode/rerank calls move to the API.
USE_HF_INFERENCE_API = os.getenv("USE_HF_INFERENCE_API", "false").lower() == "true"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GRAPH_EXTRACTION_MODEL_NAME = "llama-3.1-8b-instant"

EMBED_BATCH_SIZE = 16
CHROMA_INSERT_BATCH = 100

# Run with a SINGLE uvicorn/gunicorn worker in production. Each extra worker
# loads its own full copy of the embedding + reranker models — 2 workers
# roughly doubles RAM use for no throughput benefit on a CPU-only box this
# size. Scale by upgrading the RAM plan, not by adding workers.
WEB_CONCURRENCY = int(os.getenv("WEB_CONCURRENCY", "1"))

# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
COLLECTION_NAME = "liver_diseases"

# --------------------------------------------------------------------------
# Retrieval parameters
# --------------------------------------------------------------------------
CANDIDATE_K = 30   # pool size pulled from each retriever (semantic / BM25)
RRF_K = 60         # RRF constant
RRF_TOP_K = 8      # PRODUCTION: candidates kept after RRF fusion, before reranking
                   # (used by rag/test_search.py, and therefore by generate.py's
                   # retrieve_chunks() and app.py's /search, /ask endpoints)
EVAL_RRF_TOP_K = 20  # EVAL ONLY: wider candidate pool used by eval/evaluate_retrieval.py
                     # so the offline metrics stress-test more of the RRF ranking than
                     # production actually keeps. results.json reflects THIS value, not
                     # RRF_TOP_K above — keep that in mind when reading hit_rate/precision.
FINAL_TOP_K = 5    # final results returned after reranking

KEYWORD_COVERAGE_THRESHOLD = 0.6  # eval: fraction of expected_keywords required

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 80

# --------------------------------------------------------------------------
# Generation (rag/generate.py — the 6-step grounded-answer chain)
# --------------------------------------------------------------------------
GENERATION_MODEL_NAME = "openai/gpt-oss-120b"
# NOTE: llama-3.3-70b-versatile / llama-3.1-8b-instant are deprecated on Groq
# (announced 2026-06-17) — do not switch back to them, calls will 404.

RETRIEVAL_TOP_K = 5          # default number of chunks answer_question() retrieves
                              # (currently same value as FINAL_TOP_K above, but kept
                              # separate since they control different callers)
GRAPH_FACTS_LIMIT = 15
LOW_CONFIDENCE_THRESHOLD = 0.5
SCOPE_CONFIDENCE_FLOOR = 0.3
GENERATION_MAX_RETRIES = 3
GENERATION_RETRY_BACKOFF_S = 1.5
