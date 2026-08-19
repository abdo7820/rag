"""
app.py

FastAPI wrapper around the Liver RAG pipeline.

Production architecture:

    FastAPI
       |
       +--> Hybrid Retrieval
       |       |
       |       +--> Hugging Face Embedding API
       |       +--> BM25
       |       +--> RRF
       |       +--> Jina Reranker API
       |
       +--> Neo4j Graph
       |
       +--> Groq Generation

Railway is configured for low-RAM deployment.

IMPORTANT:
No API keys are stored in this file.
All secrets come from environment variables.
"""

import os
import pathlib
import sys

# --------------------------------------------------------------------------
# Prevent TensorFlow from being loaded accidentally.
# --------------------------------------------------------------------------

os.environ.setdefault(
    "USE_TF",
    "0",
)

# Do NOT force local Torch usage in production.
#
# The production pipeline uses remote APIs for embeddings/reranking.
#
# Local models are loaded only when the configuration explicitly requires
# them.

if os.getenv(
    "USE_HF_INFERENCE_API",
    "false",
).strip().lower() != "true":

    os.environ.setdefault(
        "USE_TORCH",
        "1",
    )


# --------------------------------------------------------------------------
# Standard library
# --------------------------------------------------------------------------

from contextlib import asynccontextmanager
from typing import Optional


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------------

from fastapi import (
    FastAPI,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from pydantic import (
    BaseModel,
    Field,
)


# ==========================================================================
# Project path
# ==========================================================================

BASE_DIR = pathlib.Path(
    __file__
).resolve().parent


PROJECT_ROOT = None


for candidate in (
    BASE_DIR,
    BASE_DIR.parent,
):

    rag_dir = candidate / "rag"

    if rag_dir.exists():

        sys.path.insert(
            0,
            str(rag_dir),
        )

        sys.path.insert(
            0,
            str(candidate),
        )

        PROJECT_ROOT = candidate

        break


if PROJECT_ROOT is None:

    raise RuntimeError(
        "Could not find a 'rag/' folder next to app.py "
        "or its parent. Place app.py at the project root."
    )


# ==========================================================================
# Third-party imports
# ==========================================================================

from groq import Groq
from neo4j import GraphDatabase


# ==========================================================================
# Project imports
# ==========================================================================

from config import (
    IS_PRODUCTION,
    ENVIRONMENT,
    USE_HF_INFERENCE_API,
    USE_JINA_RERANKER_API,
    HF_API_TOKEN,
    JINA_API_KEY,
    JINA_RERANKER_MODEL,
    WEB_CONCURRENCY,
    RETRIEVAL_TOP_K,
)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

from test_search import (
    semantic_search,
    bm25_search,
    reciprocal_rank_fusion,
    rerank_results,
)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

from generate import (
    answer_question,
    retrieve_graph_facts,
    get_neo4j_driver,
)


# ==========================================================================
# Application state
# ==========================================================================

state: dict = {
    "groq_client": None,
    "neo4j_driver": None,
}


# ==========================================================================
# Lifespan
# ==========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # ----------------------------------------------------------------------
    # Groq
    # ----------------------------------------------------------------------

    groq_key = os.getenv(
        "GROQ_API_KEY"
    )

    if groq_key:

        groq_key = groq_key.strip()

        state["groq_client"] = Groq(
            api_key=groq_key
        )

    else:

        state["groq_client"] = None


    # ----------------------------------------------------------------------
    # Neo4j
    # ----------------------------------------------------------------------

    try:

        state["neo4j_driver"] = (
            get_neo4j_driver()
        )

    except Exception as exc:

        print(
            f"[WARN] Neo4j initialization failed: {exc}"
        )

        state["neo4j_driver"] = None


    # ----------------------------------------------------------------------
    # Startup information
    # ----------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Liver RAG API started"
    )

    print(
        "=" * 70
    )

    print(
        f"Environment              : {ENVIRONMENT}"
    )

    print(
        f"Production               : {IS_PRODUCTION}"
    )

    print(
        f"HF Embedding API         : "
        f"{USE_HF_INFERENCE_API}"
    )

    print(
        f"Jina Reranker API        : "
        f"{USE_JINA_RERANKER_API}"
    )

    if USE_JINA_RERANKER_API:

        print(
            f"Jina Reranker Model      : "
            f"{JINA_RERANKER_MODEL}"
        )

    print(
        f"HF Token configured      : "
        f"{bool(HF_API_TOKEN)}"
    )

    print(
        f"Jina API Key configured  : "
        f"{bool(JINA_API_KEY)}"
    )

    print(
        f"Groq configured          : "
        f"{state['groq_client'] is not None}"
    )

    print(
        f"Neo4j configured         : "
        f"{state['neo4j_driver'] is not None}"
    )

    print(
        f"Web concurrency          : "
        f"{WEB_CONCURRENCY}"
    )

    print(
        "=" * 70
        + "\n"
    )


    # ----------------------------------------------------------------------
    # Application running
    # ----------------------------------------------------------------------

    yield


    # ----------------------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------------------

    if (
        state["neo4j_driver"]
        is not None
    ):

        try:

            state[
                "neo4j_driver"
            ].close()

        except Exception as exc:

            print(
                f"[WARN] Neo4j shutdown error: {exc}"
            )


# ==========================================================================
# FastAPI application
# ==========================================================================

app = FastAPI(
    title="Liver Diseases RAG API",
    description=(
        "Hybrid semantic + BM25 + RRF retrieval, "
        "Jina reranking, Neo4j graph context, "
        "and grounded Q&A over the hepatology paper."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ==========================================================================
# CORS
# ==========================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# Request / Response models
# ==========================================================================

class SearchRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        example="What causes liver cirrhosis?",
    )

    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description=(
            "Final number of results after reranking."
        ),
    )

    rerank: bool = Field(
        True,
        description=(
            "Run the configured reranking stage."
        ),
    )


class ChunkResult(BaseModel):

    rank: int

    chunk_id: int | str | None

    score: float

    section: Optional[str] = None

    page_start: Optional[int] = None

    page_end: Optional[int] = None

    doi: Optional[str] = None

    source: Optional[str] = None

    retrieval_method: Optional[str] = None

    text: str


class SearchResponse(BaseModel):

    query: str

    reranked: bool

    results: list[ChunkResult]


class AskRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        example="What causes liver cirrhosis?",
    )

    top_k: int = Field(
        RETRIEVAL_TOP_K,
        ge=1,
        le=20,
    )


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

    hf_embedding_api: bool

    jina_reranker_api: bool

    jina_reranker_model: Optional[str] = None


# ==========================================================================
# Health endpoint
# ==========================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
)
def health():

    return HealthResponse(

        status="ok",

        environment=ENVIRONMENT,

        groq_configured=(
            state["groq_client"]
            is not None
        ),

        neo4j_configured=(
            state["neo4j_driver"]
            is not None
        ),

        vector_db_found=(
            PROJECT_ROOT / "vectorDB"
        ).exists(),

        bm25_index_found=(
            PROJECT_ROOT
            / "data"
            / "bm25_index.pkl"
        ).exists(),

        hf_embedding_api=(
            USE_HF_INFERENCE_API
        ),

        jina_reranker_api=(
            USE_JINA_RERANKER_API
        ),

        jina_reranker_model=(
            JINA_RERANKER_MODEL
            if USE_JINA_RERANKER_API
            else None
        ),
    )


# ==========================================================================
# Retrieval helper
# ==========================================================================

def _retrieval_method(
    r: dict,
) -> str:

    has_semantic = (
        r.get(
            "semantic_rank"
        )
        is not None
    )

    has_bm25 = (
        r.get(
            "bm25_rank"
        )
        is not None
    )

    if (
        has_semantic
        and has_bm25
    ):

        return "both"

    if has_semantic:

        return "semantic"

    return "bm25"


# ==========================================================================
# /search
# ==========================================================================

@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["retrieval"],
)
def search(
    req: SearchRequest,
):

    """
    Hybrid retrieval only.

    Pipeline:

        Semantic Search
              +
            BM25
              ↓
            RRF
              ↓
       Jina Reranker API
              ↓
           Top K

    No LLM generation is performed here.
    """

    try:

        # --------------------------------------------------------------
        # Semantic retrieval
        # --------------------------------------------------------------

        semantic_results = (
            semantic_search(
                req.query
            )
        )


        # --------------------------------------------------------------
        # BM25 retrieval
        # --------------------------------------------------------------

        bm25_results = (
            bm25_search(
                req.query
            )
        )


        # --------------------------------------------------------------
        # RRF
        # --------------------------------------------------------------

        fused = (
            reciprocal_rank_fusion(
                semantic_results,
                bm25_results,
            )
        )


        # --------------------------------------------------------------
        # Reranking
        # --------------------------------------------------------------

        if req.rerank:

            final = (
                rerank_results(
                    req.query,
                    fused,
                )
            )

            final = final[
                :req.top_k
            ]

            score_key = (
                "reranker_score"
            )

        else:

            final = fused[
                :req.top_k
            ]

            score_key = (
                "rrf_score"
            )


        # --------------------------------------------------------------
        # Format API response
        # --------------------------------------------------------------

        results = []


        for rank, r in enumerate(
            final,
            start=1,
        ):

            meta = (
                r.get(
                    "metadata",
                    {},
                )
                or {}
            )


            page_start = (
                meta.get(
                    "page_start"
                )
            )

            page_end = (
                meta.get(
                    "page_end"
                )
            )


            if page_start in (
                None,
                -1,
            ):

                page_start = None


            if page_end in (
                None,
                -1,
            ):

                page_end = None


            results.append(
                ChunkResult(

                    rank=rank,

                    chunk_id=r.get(
                        "chunk_id"
                    ),

                    score=float(
                        r.get(
                            score_key,
                            0.0,
                        )
                    ),

                    section=meta.get(
                        "section"
                    ),

                    page_start=page_start,

                    page_end=page_end,

                    doi=meta.get(
                        "doi"
                    ),

                    source=meta.get(
                        "source"
                    ),

                    retrieval_method=(
                        _retrieval_method(r)
                    ),

                    text=r[
                        "text"
                    ],
                )
            )


        return SearchResponse(

            query=req.query,

            reranked=req.rerank,

            results=results,
        )


    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


    except Exception as exc:

        print(
            f"[ERROR] /search failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Search failed: {exc}"
            ),
        ) from exc


# ==========================================================================
# /ask
# ==========================================================================

@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["generation"],
)
def ask(
    req: AskRequest,
):

    """
    Runs the grounded-answer pipeline.

    IMPORTANT:
    The actual retrieval/generation implementation lives in
    rag/generate.py.

    This endpoint does not load local embedding/reranker models itself.
    """

    # ----------------------------------------------------------------------
    # Groq
    # ----------------------------------------------------------------------

    if (
        state["groq_client"]
        is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY is not configured."
            ),
        )


    # ----------------------------------------------------------------------
    # Neo4j
    # ----------------------------------------------------------------------

    if (
        state["neo4j_driver"]
        is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Neo4j is not configured."
            ),
        )


    try:

        answer = answer_question(

            state[
                "groq_client"
            ],

            state[
                "neo4j_driver"
            ],

            req.question,

            top_k=req.top_k,
        )


        return AskResponse(

            question=req.question,

            answer=answer,
        )


    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


    except Exception as exc:

        print(
            f"[ERROR] /ask failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Answer generation failed: {exc}"
            ),
        ) from exc


# ==========================================================================
# /graph/{entity_name}
# ==========================================================================

@app.get(
    "/graph/{entity_name}",
    response_model=GraphResponse,
    tags=["graph"],
)
def graph_lookup(
    entity_name: str,
):

    if (
        state["neo4j_driver"]
        is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Neo4j is not configured."
            ),
        )


    try:

        facts_raw = (
            retrieve_graph_facts(
                state[
                    "neo4j_driver"
                ],

                entity_name,

                limit=50,
            )
        )


        facts = [

            GraphFact(

                source=f[
                    "source"
                ],

                relation=f[
                    "relation"
                ],

                target=f[
                    "target"
                ],

                section=f.get(
                    "section"
                ),

                page=f.get(
                    "page"
                ),
            )

            for f in facts_raw
        ]


        return GraphResponse(

            entity=entity_name,

            facts=facts,
        )


    except Exception as exc:

        print(
            f"[ERROR] /graph failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Graph lookup failed: {exc}"
            ),
        ) from exc


# ==========================================================================
# Local development
# ==========================================================================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        "app:app",

        host="127.0.0.1",

        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),

        reload=True,
    )
