"""
app.py

FastAPI wrapper around the RAG pipeline. Reads all tunables from config.py.

Install:
    pip install -r requirements.txt

Run (local):
    uvicorn app:app --reload --port 8000

Run (production, e.g. Railway):
    See Procfile — single worker, binds to $PORT.

Endpoints:
    GET  /health
    POST /search   {"query": "...", "top_k": 5, "rerank": true}
    POST /ask      {"question": "..."}
    GET  /graph/{entity_name}
"""
import os
import pathlib
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = pathlib.Path(__file__).resolve().parent
for candidate in (BASE_DIR, BASE_DIR.parent):
    rag_dir = candidate / "rag"
    if rag_dir.exists():
        sys.path.insert(0, str(rag_dir))
        sys.path.insert(0, str(candidate))
        PROJECT_ROOT = candidate
        break
else:
    raise RuntimeError(
        "Could not find a 'rag/' folder next to app.py or its parent. "
        "Place app.py at the project root (next to rag/, models/, data/)."
    )

from groq import Groq  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402

from config import IS_PRODUCTION  # noqa: E402
from test_search import semantic_search, bm25_search, reciprocal_rank_fusion, rerank_results  # noqa: E402
from generate import (  # noqa: E402
    answer_question,
    retrieve_graph_facts,
    get_neo4j_driver,
    RETRIEVAL_TOP_K,
)

state: dict = {"groq_client": None, "neo4j_driver": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("GROQ_API_KEY")
    state["groq_client"] = Groq(api_key=api_key) if api_key else None
    state["neo4j_driver"] = get_neo4j_driver()
    yield
    if state["neo4j_driver"] is not None:
        state["neo4j_driver"].close()


app = FastAPI(
    title="Liver Diseases RAG API",
    description="Hybrid (semantic + BM25 + graph) retrieval and grounded Q&A over the hepatology paper.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: wide open for local/demo use. Tighten allow_origins to your actual
# frontend domain before a real public production deployment.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, example="What causes liver cirrhosis?")
    top_k: int = Field(5, ge=1, le=20, description="Final number of results after reranking")
    rerank: bool = Field(True, description="Whether to run the cross-encoder rerank stage")


class ChunkResult(BaseModel):
    rank: int
    chunk_id: int | str | None
    score: float
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    doi: Optional[str] = None
    source: Optional[str] = None
    retrieval_method: Optional[str] = None  # "semantic" | "bm25" | "both"
    text: str


class SearchResponse(BaseModel):
    query: str
    reranked: bool
    results: list[ChunkResult]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, example="What causes liver cirrhosis?")
    top_k: int = Field(RETRIEVAL_TOP_K, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str


class GraphFact(BaseModel):
    source: str
    relation: str
    target: str
    section: Optional[str] = None
    page: Optional[int] = None


class GraphResponse(BaseModel):
    entity: str
    facts: list[GraphFact]


class HealthResponse(BaseModel):
    status: str
    environment: str
    groq_configured: bool
    neo4j_configured: bool
    vector_db_found: bool
    bm25_index_found: bool


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        environment="production" if IS_PRODUCTION else "development",
        groq_configured=state["groq_client"] is not None,
        neo4j_configured=state["neo4j_driver"] is not None,
        vector_db_found=(PROJECT_ROOT / "vectorDB").exists(),
        bm25_index_found=(PROJECT_ROOT / "data" / "bm25_index.pkl").exists(),
    )


def _retrieval_method(r: dict) -> str:
    has_semantic = r.get("semantic_rank") is not None
    has_bm25 = r.get("bm25_rank") is not None
    if has_semantic and has_bm25:
        return "both"
    return "semantic" if has_semantic else "bm25"


@app.post("/search", response_model=SearchResponse, tags=["retrieval"])
def search(req: SearchRequest):
    """Hybrid retrieval only (no LLM answer): semantic + BM25 -> RRF fusion,
    optionally followed by cross-encoder reranking."""
    try:
        semantic_results = semantic_search(req.query)
        bm25_results = bm25_search(req.query)
        fused = reciprocal_rank_fusion(semantic_results, bm25_results)

        if req.rerank:
            final = rerank_results(req.query, fused)[: req.top_k]
            score_key = "reranker_score"
        else:
            final = fused[: req.top_k]
            score_key = "rrf_score"

        results = []
        for rank, r in enumerate(final, start=1):
            meta = r.get("metadata", {}) or {}
            results.append(
                ChunkResult(
                    rank=rank,
                    chunk_id=r.get("chunk_id"),
                    score=float(r.get(score_key, 0.0)),
                    section=meta.get("section"),
                    page_start=meta.get("page_start") if meta.get("page_start") not in (None, -1) else None,
                    page_end=meta.get("page_end") if meta.get("page_end") not in (None, -1) else None,
                    doi=meta.get("doi"),
                    source=meta.get("source"),
                    retrieval_method=_retrieval_method(r),
                    text=r["text"],
                )
            )

        return SearchResponse(query=req.query, reranked=req.rerank, results=results)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


@app.post("/ask", response_model=AskResponse, tags=["generation"])
def ask(req: AskRequest):
    """Runs the full 6-step grounded-answer chain and returns a cited answer."""
    if state["groq_client"] is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set on the server (.env).")

    try:
        answer = answer_question(
            state["groq_client"], state["neo4j_driver"], req.question, top_k=req.top_k
        )
        return AskResponse(question=req.question, answer=answer)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}") from exc


@app.get("/graph/{entity_name}", response_model=GraphResponse, tags=["graph"])
def graph_lookup(entity_name: str):
    if state["neo4j_driver"] is None:
        raise HTTPException(status_code=503, detail="Neo4j is not configured on the server (.env).")

    try:
        facts_raw = retrieve_graph_facts(state["neo4j_driver"], entity_name, limit=50)
        facts = [
            GraphFact(
                source=f["source"], relation=f["relation"], target=f["target"],
                section=f.get("section"), page=f.get("page"),
            )
            for f in facts_raw
        ]
        return GraphResponse(entity=entity_name, facts=facts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph lookup failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
