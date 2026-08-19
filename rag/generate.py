"""
rag/generate.py

6-step grounded-answer chain for the Liver RAG system.

Pipeline:

    1. Guardrail
    2. Query rewrite
    3. Hybrid retrieval
           Semantic
              +
           BM25
              ↓
             RRF
              ↓
        Jina Reranker API
    4. Scope check
    5. Draft answer
    6. Verify + finalize

Production architecture:

    Railway
       |
       +-- Hugging Face Inference API
       |       └── query embeddings
       |
       +-- Jina API
       |       └── reranking
       |
       +-- Chroma + BM25
       |
       +-- Neo4j
       |
       └-- Groq

IMPORTANT:
This file does NOT load torch or local embedding/reranker models.
Local model loading, when explicitly requested, is handled by
rag/test_search.py.
"""

import os
import pathlib
import re
import sys
import time


# ==========================================================================
# Environment
# ==========================================================================

from dotenv import load_dotenv

load_dotenv()


# ==========================================================================
# Project path
# ==========================================================================

BASE_DIR = pathlib.Path(
    __file__
).resolve().parent.parent

sys.path.insert(
    0,
    str(BASE_DIR / "rag"),
)

sys.path.insert(
    0,
    str(BASE_DIR),
)


# ==========================================================================
# Third-party imports
# ==========================================================================

from groq import Groq
from neo4j import GraphDatabase


# ==========================================================================
# Retrieval imports
# ==========================================================================

from test_search import (
    semantic_search,
    bm25_search,
    reciprocal_rank_fusion,
    rerank_results,
)


# ==========================================================================
# Configuration
# ==========================================================================

from config import (
    GENERATION_MODEL_NAME as MODEL_NAME,
    LOW_CONFIDENCE_THRESHOLD,
    SCOPE_CONFIDENCE_FLOOR,
    GRAPH_FACTS_LIMIT,
    RETRIEVAL_TOP_K,
    GENERATION_MAX_RETRIES,
    GENERATION_RETRY_BACKOFF_S,
    USE_HF_INFERENCE_API,
    USE_JINA_RERANKER_API,
    JINA_RERANKER_MODEL,
)


# ==========================================================================
# Startup configuration
# ==========================================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Liver RAG generation pipeline"
)

print(
    "=" * 70
)

print(
    f"HF embedding API : "
    f"{USE_HF_INFERENCE_API}"
)

print(
    f"Jina reranker API: "
    f"{USE_JINA_RERANKER_API}"
)

if USE_JINA_RERANKER_API:

    print(
        f"Jina model       : "
        f"{JINA_RERANKER_MODEL}"
    )

print(
    f"Generation model : "
    f"{MODEL_NAME}"
)

print(
    "=" * 70
    + "\n"
)


# ==========================================================================
# Constants
# ==========================================================================

NOT_FOUND_TEXT = (
    "I don't know based on the available sources."
)


OUT_OF_SCOPE_TEXT = (
    "That's outside what this hepatology paper covers, "
    "so I can't answer it from these sources."
)


UNVERIFIED_TEXT = (
    "I found some potentially relevant material, but I "
    "couldn't confirm the answer was fully backed by a "
    "citation I could verify, so I'm not showing it rather "
    "than risk giving you an unsupported claim. Please try "
    "rephrasing the question or asking again."
)


DISCLAIMER = (
    "\n\n---\n"
    "*This is educational information drawn from a research "
    "paper, not medical advice. Please consult a qualified "
    "healthcare professional for any personal medical decisions.*"
)


# ==========================================================================
# Citation parser
# ==========================================================================

CITATION_RE = re.compile(
    r"\[(Graph source|Source):\s*([^,\]]+?)"
    r",\s*p\.\s*([\w\-]+)\s*\]"
)


# ==========================================================================
# Generation error
# ==========================================================================

class GenerationError(RuntimeError):
    """
    Raised when a required generation step fails after retries.

    We never fall back to an unverified answer.
    """

    pass


# ==========================================================================
# Step 1 — Guardrail
# ==========================================================================

BLOCKED_PATTERNS = re.compile(
    r"\b("
    r"dosage|"
    r"dose|"
    r"prescri\w*|"
    r"should i take|"
    r"how much .*(should|do i) take|"
    r"is it safe for me|"
    r"diagnose me|"
    r"do i have"
    r")\b",
    re.IGNORECASE,
)


def is_blocked(
    question: str,
) -> bool:

    return bool(
        BLOCKED_PATTERNS.search(
            question
        )
    )


# ==========================================================================
# Groq helper
# ==========================================================================

def _create_completion(
    client: Groq,
    **kwargs,
):

    try:

        return (
            client
            .chat
            .completions
            .create(**kwargs)
        )

    except TypeError as exc:

        if "reasoning_effort" in str(exc):

            kwargs.pop(
                "reasoning_effort",
                None,
            )

            return _create_completion(
                client,
                **kwargs,
            )

        if "max_completion_tokens" in str(exc):

            kwargs["max_tokens"] = (
                kwargs.pop(
                    "max_completion_tokens"
                )
            )

            return _create_completion(
                client,
                **kwargs,
            )

        raise


def _create_completion_with_retry(
    client: Groq,
    *,
    step_name: str,
    **kwargs,
) -> str:

    last_error = None

    for attempt in range(
        1,
        GENERATION_MAX_RETRIES + 1,
    ):

        try:

            response = (
                _create_completion(
                    client,
                    **kwargs,
                )
            )

            text = (
                response
                .choices[0]
                .message
                .content
                or ""
            ).strip()

            if text:

                return text

            last_error = RuntimeError(
                f"{step_name}: "
                "model returned an empty response"
            )

        except Exception as exc:

            last_error = exc

            print(
                f"[DEBUG] {step_name} "
                f"attempt {attempt}/"
                f"{GENERATION_MAX_RETRIES} "
                f"failed: {exc!r}"
            )

        if (
            attempt
            < GENERATION_MAX_RETRIES
        ):

            time.sleep(
                GENERATION_RETRY_BACKOFF_S
                * attempt
            )

    raise GenerationError(
        f"{step_name} failed after "
        f"{GENERATION_MAX_RETRIES} attempts: "
        f"{last_error}"
    ) from last_error


# ==========================================================================
# Step 2 — Query rewrite
# ==========================================================================

REWRITE_PROMPT = """
Rewrite the question below into a focused search query
for retrieving passages from a hepatology (liver disease)
research paper.

Rules:

- Expand medical abbreviations to their full form alongside
  the abbreviation.

- Add closely related medical synonyms if they would help
  retrieval.

- Keep it short.

- Output a search query, not an answer.

- Do not add commentary.

Question:
{question}

Search query:
"""


def rewrite_query(
    client: Groq,
    question: str,
) -> str:

    try:

        response = (
            client
            .chat
            .completions
            .create(

                model=MODEL_NAME,

                messages=[
                    {
                        "role": "user",
                        "content":
                            REWRITE_PROMPT.format(
                                question=question
                            ),
                    }
                ],

                temperature=0,

                reasoning_effort="low",

                max_completion_tokens=200,
            )
        )

        rewritten = (
            response
            .choices[0]
            .message
            .content
            or ""
        ).strip()

        return (
            rewritten
            if rewritten
            else question
        )

    except Exception as exc:

        print(
            f"[WARN] Query rewrite failed: "
            f"{exc}"
        )

        return question


# ==========================================================================
# Step 3 — Hybrid retrieval
# ==========================================================================

def retrieve_chunks(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:

    """
    Hybrid retrieval:

        Semantic
           +
         BM25
           ↓
          RRF
           ↓
       Reranking
           ↓
         Top K

    The actual reranker backend is selected by config:

        Production:
            Jina API

        Local:
            CrossEncoder
    """

    print(
        "\n=== RETRIEVAL ==="
    )

    # ----------------------------------------------------------------------
    # Semantic
    # ----------------------------------------------------------------------

    semantic_results = (
        semantic_search(
            query
        )
    )

    print(
        f"[RETRIEVAL] Semantic results: "
        f"{len(semantic_results)}"
    )


    # ----------------------------------------------------------------------
    # BM25
    # ----------------------------------------------------------------------

    bm25_results = (
        bm25_search(
            query
        )
    )

    print(
        f"[RETRIEVAL] BM25 results: "
        f"{len(bm25_results)}"
    )


    # ----------------------------------------------------------------------
    # RRF
    # ----------------------------------------------------------------------

    fused = (
        reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
        )
    )

    print(
        f"[RETRIEVAL] RRF candidates: "
        f"{len(fused)}"
    )


    # ----------------------------------------------------------------------
    # Reranking
    # ----------------------------------------------------------------------

    reranked = (
        rerank_results(
            query,
            fused,
        )
    )

    print(
        f"[RETRIEVAL] Reranked results: "
        f"{len(reranked)}"
    )


    # ----------------------------------------------------------------------
    # Final top K
    # ----------------------------------------------------------------------

    final = reranked[
        :top_k
    ]


    # ----------------------------------------------------------------------
    # Debug ranking
    # ----------------------------------------------------------------------

    for index, chunk in enumerate(
        final,
        start=1,
    ):

        print(
            f"[RETRIEVAL] #{index} "
            f"chunk={chunk.get('chunk_id')} "
            f"reranker="
            f"{chunk.get('reranker_score', 0.0):.4f} "
            f"rrf="
            f"{chunk.get('rrf_score', 0.0):.6f}"
        )

    return final


# ==========================================================================
# Document context
# ==========================================================================

def build_chunk_context(
    chunks: list[dict],
) -> str:

    blocks = []

    for c in chunks:

        meta = (
            c.get(
                "metadata",
                {},
            )
            or {}
        )

        section = (
            meta.get(
                "section"
            )
            or "Unknown"
        )

        page = (
            meta.get(
                "page_start"
            )

            if meta.get(
                "page_start"
            )
            not in (
                None,
                -1,
            )

            else "?"
        )

        blocks.append(
            f"[Source: {section}, p.{page}]\n"
            f"{c['text']}"
        )

    return (
        "\n\n".join(
            blocks
        )
    )


# ==========================================================================
# Neo4j
# ==========================================================================

def get_neo4j_driver():

    uri = os.getenv(
        "NEO4J_URI"
    )

    user = os.getenv(
        "NEO4J_USER"
    )

    password = os.getenv(
        "NEO4J_PASSWORD"
    )

    if not all(
        [
            uri,
            user,
            password,
        ]
    ):

        return None


    driver = GraphDatabase.driver(
        uri,
        auth=(
            user,
            password,
        ),
    )


    try:

        driver.verify_connectivity()

    except Exception as exc:

        print(
            f"[WARN] Neo4j configured but "
            f"unreachable: {exc}"
        )

        print(
            "[WARN] Continuing without graph facts."
        )

        driver.close()

        return None


    return driver


# ==========================================================================
# Graph helpers
# ==========================================================================

def _dedupe_graph_facts(
    facts: list[dict],
) -> list[dict]:

    seen = set()

    deduped = []

    for f in facts:

        key = (
            f.get("source"),
            f.get("relation"),
            f.get("target"),
            f.get("section"),
            f.get("page"),
        )

        if key in seen:

            continue

        seen.add(
            key
        )

        deduped.append(
            f
        )

    return deduped


_all_entity_names: list[str] | None = None


def _get_all_entity_names(
    driver,
    database: str,
) -> list[str]:

    global _all_entity_names

    if _all_entity_names is None:

        with driver.session(
            database=database
        ) as session:

            result = session.run(
                """
                MATCH (e:Entity)
                WHERE size(e.name) > 2
                RETURN DISTINCT e.name AS name
                """
            )

            _all_entity_names = [
                record["name"]
                for record in result
            ]

    return _all_entity_names


def _find_mentioned_entities(
    question: str,
    all_names: list[str],
) -> list[str]:

    question_lower = (
        question.lower()
    )

    matched = []

    for name in all_names:

        pattern = (
            r"\b"
            + re.escape(
                name.lower()
            )
            + r"\b"
        )

        if re.search(
            pattern,
            question_lower,
        ):

            matched.append(
                name
            )

    return matched


def retrieve_graph_facts(
    driver,
    original_question: str,
    limit: int = GRAPH_FACTS_LIMIT,
) -> list[dict]:

    if driver is None:

        return []


    database = os.getenv(
        "NEO4J_DATABASE",
        "neo4j",
    )


    try:

        all_names = (
            _get_all_entity_names(
                driver,
                database,
            )
        )

        names = (
            _find_mentioned_entities(
                original_question,
                all_names,
            )
        )


        if not names:

            return []


        with driver.session(
            database=database
        ) as session:

            result = session.run(

                """
                MATCH
                    (a:Entity)-[r:RELATION]->(b:Entity)

                WHERE
                    a.name IN $names
                    OR
                    b.name IN $names

                RETURN
                    a.name AS source,
                    r.relation AS relation,
                    b.name AS target,
                    r.section AS section,
                    r.page_start AS page

                LIMIT $limit
                """,

                names=names,

                limit=limit,
            )


            facts = [
                dict(record)
                for record in result
            ]


            return _dedupe_graph_facts(
                facts
            )


    except Exception as exc:

        print(
            f"[WARN] Neo4j query failed: "
            f"{exc}"
        )

        return []


def build_graph_context(
    facts: list[dict],
) -> str:

    if not facts:

        return ""


    lines = []


    for f in facts:

        section = (
            f.get("section")
            or "Unknown"
        )

        page = (
            f.get("page")
            or "?"
        )


        lines.append(
            "[Graph source: "
            f"{section}, p.{page}] "
            f"{f['source']} "
            f"-[{f['relation']}]-> "
            f"{f['target']}"
        )


    return "\n".join(
        lines
    )


def build_full_context(
    chunks: list[dict],
    graph_facts: list[dict],
) -> str:

    parts = []


    chunk_context = (
        build_chunk_context(
            chunks
        )
    )


    if chunk_context:

        parts.append(
            "### DOCUMENT EXCERPTS\n\n"
            + chunk_context
        )


    graph_context = (
        build_graph_context(
            graph_facts
        )
    )


    if graph_context:

        parts.append(
            "### KNOWLEDGE GRAPH FACTS\n\n"
            + graph_context
        )


    return (
        "\n\n".join(parts)
        if parts
        else "(no evidence retrieved)"
    )


# ==========================================================================
# Step 4 — Scope check
# ==========================================================================

SCOPE_PROMPT = """
You will be shown a QUESTION and EVIDENCE excerpts
retrieved from a hepatology research paper.

Think briefly about whether:

1. The question is related to hepatology/liver disease.
2. The evidence is at least topically related.

The evidence does NOT need to completely answer the question.

Respond exactly:

Reasoning: <one short sentence>
Verdict: YES or NO

QUESTION:
{question}

EVIDENCE:
{context}
"""


def classify_scope(
    client: Groq,
    question: str,
    context: str,
) -> bool:

    try:

        response = (
            _create_completion(
                client,

                model=MODEL_NAME,

                messages=[
                    {
                        "role": "user",
                        "content":
                            SCOPE_PROMPT.format(
                                question=question,
                                context=context,
                            ),
                    }
                ],

                temperature=0,

                reasoning_effort="low",

                max_completion_tokens=150,
            )
        )


        text = (
            response
            .choices[0]
            .message
            .content
            or ""
        ).strip().upper()


        verdict_line = next(
            (
                line
                for line in text.splitlines()
                if "VERDICT" in line
            ),
            text,
        )


        return (
            "YES"
            in verdict_line
        )


    except Exception as exc:

        print(
            f"[WARN] Scope check failed: "
            f"{exc}"
        )

        return True


def is_in_scope(
    client: Groq,
    question: str,
    context: str,
    top_reranker_score: float,
) -> bool:

    retrieval_is_weak = (
        top_reranker_score
        < SCOPE_CONFIDENCE_FLOOR
    )


    if not retrieval_is_weak:

        return True


    llm_says_in_scope = (
        classify_scope(
            client,
            question,
            context,
        )
    )


    return not (
        not llm_says_in_scope
        and retrieval_is_weak
    )


# ==========================================================================
# Step 5 — Draft answer
# ==========================================================================

DRAFT_PROMPT = """
You are a hepatology research assistant answering
questions strictly from the material provided below.

Evidence types:

1. DOCUMENT EXCERPTS
   Passages retrieved from the research paper.

2. KNOWLEDGE GRAPH FACTS
   Entity relationships extracted from the same paper.

Rules:

1. Read ALL evidence before answering.

2. Only state information directly supported by evidence.

3. Never use outside knowledge.

4. After every factual claim, add a citation using EXACTLY:

   [Source: <section>, p.<page>]

   or

   [Graph source: <section>, p.<page>]

5. Never invent citations.

6. Never invent page numbers.

7. If evidence only partially answers the question,
   answer only the supported part.

8. If evidence does not answer the question, output
   exactly:

   {not_found}

9. Prefer both document evidence and graph evidence
   when both are relevant.

10. Be concise.

11. Do not add medical disclaimers.
    The system adds them separately.

EVIDENCE:

{context}
""".replace(
    "{not_found}",
    NOT_FOUND_TEXT,
)


def draft_answer(
    client: Groq,
    query: str,
    context: str,
) -> str:

    system_prompt = (
        DRAFT_PROMPT.format(
            context=context
        )
    )


    return (
        _create_completion_with_retry(
            client,

            step_name="draft_answer",

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content":
                        system_prompt,
                },
                {
                    "role": "user",
                    "content":
                        query,
                },
            ],

            temperature=0,

            reasoning_effort="low",

            max_completion_tokens=2000,
        )
    )


# ==========================================================================
# Step 6 — Verify
# ==========================================================================

VERIFY_PROMPT = """
You are a strict fact-checker.

QUESTION:
{question}

EVIDENCE:
{context}

DRAFT ANSWER:
{draft}

Check the draft claim by claim.

Rules:

- Every factual claim must be supported by the evidence.

- Every citation must reference evidence that actually
  appears above.

- Every citation must contain both section and page.

- Correct unsupported claims.

- Remove unsupported claims.

- Keep valid claims and citations.

- If nothing remains supported, output exactly:

  {not_found}

- Output ONLY the final answer.

- Do not explain what you changed.
""".replace(
    "{not_found}",
    NOT_FOUND_TEXT,
)


def verify_answer(
    client: Groq,
    query: str,
    context: str,
    draft: str,
) -> str:

    return (
        _create_completion_with_retry(

            client,

            step_name="verify_answer",

            model=MODEL_NAME,

            messages=[
                {
                    "role": "user",
                    "content":
                        VERIFY_PROMPT.format(
                            question=query,
                            context=context,
                            draft=draft,
                        ),
                }
            ],

            temperature=0,

            reasoning_effort="low",

            max_completion_tokens=2500,
        )
    )


# ==========================================================================
# Citation validation
# ==========================================================================

def _extract_citations(
    text: str,
) -> list[tuple[str, str, str]]:

    return [
        (
            kind.strip(),
            section.strip(),
            page.strip(),
        )

        for kind, section, page
        in CITATION_RE.findall(
            text
        )
    ]


def _valid_citation_keys(
    chunks: list[dict],
    graph_facts: list[dict],
) -> set[tuple[str, str, str]]:

    valid = set()


    for c in chunks:

        meta = (
            c.get(
                "metadata",
                {},
            )
            or {}
        )


        section = str(
            meta.get(
                "section",
                "Unknown",
            )
        ).strip()


        page = str(
            meta.get(
                "page_start",
                "?",
            )
        ).strip()


        valid.add(
            (
                "Source",
                section,
                page,
            )
        )


    for f in graph_facts:

        section = str(
            f.get(
                "section"
            )
            or "Unknown"
        ).strip()


        page = str(
            f.get(
                "page"
            )
            or "?"
        ).strip()


        valid.add(
            (
                "Graph source",
                section,
                page,
            )
        )


    return valid


def has_only_verifiable_citations(
    answer: str,
    chunks: list[dict],
    graph_facts: list[dict],
) -> bool:

    citations = (
        _extract_citations(
            answer
        )
    )


    if not citations:

        return True


    valid_keys = (
        _valid_citation_keys(
            chunks,
            graph_facts,
        )
    )


    return all(
        citation in valid_keys
        for citation in citations
    )


# ==========================================================================
# Finalize
# ==========================================================================

def finalize_answer(
    answer: str,
    top_reranker_score: float,
) -> str:

    confidence_note = ""


    if (
        top_reranker_score
        < LOW_CONFIDENCE_THRESHOLD
    ):

        confidence_note = (
            "\n\n⚠️ "
            "*Low confidence retrieval — "
            "the closest match found wasn't "
            "a strong fit for this question.*"
        )


    return (
        answer
        + confidence_note
        + DISCLAIMER
    )


# ==========================================================================
# Main chain
# ==========================================================================

def answer_question(
    client: Groq,
    driver,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> str:

    # ----------------------------------------------------------------------
    # Step 1
    # ----------------------------------------------------------------------

    if is_blocked(
        query
    ):

        return (
            "I can't give personal medical advice "
            "(dosage, prescriptions, or diagnosis) "
            "— please talk to a doctor or pharmacist "
            "about that. I can share what the research "
            "paper says about liver disease topics in "
            "general, if that helps."
        )


    t_start = time.time()


    # ----------------------------------------------------------------------
    # Step 2 — Rewrite
    # ----------------------------------------------------------------------

    t0 = time.time()

    search_query = (
        rewrite_query(
            client,
            query,
        )
    )

    print(
        f"[TIMING] rewrite_query: "
        f"{time.time() - t0:.2f}s"
    )

    print(
        f"[QUERY] Original : {query}"
    )

    print(
        f"[QUERY] Rewritten: {search_query}"
    )


    # ----------------------------------------------------------------------
    # Step 3 — Retrieval
    # ----------------------------------------------------------------------

    t0 = time.time()

    chunks = (
        retrieve_chunks(
            search_query,
            top_k=top_k,
        )
    )

    print(
        f"[TIMING] retrieve_chunks "
        f"(semantic+bm25+rrf+rerank): "
        f"{time.time() - t0:.2f}s"
    )


    # ----------------------------------------------------------------------
    # Graph
    # ----------------------------------------------------------------------

    t0 = time.time()

    graph_facts = (
        retrieve_graph_facts(
            driver,
            query,
        )
    )

    print(
        f"[TIMING] retrieve_graph_facts: "
        f"{time.time() - t0:.2f}s"
    )


    print(
        "\n=== Graph facts ==="
    )


    if not graph_facts:

        print(
            "(none retrieved)"
        )

    else:

        for f in graph_facts:

            section = (
                f.get(
                    "section"
                )
                or "Unknown"
            )

            page = (
                f.get(
                    "page"
                )
                or "?"
            )

            print(
                f"{f['source']} "
                f"-[{f['relation']}]-> "
                f"{f['target']} "
                f"(section={section}, "
                f"p.{page})"
            )


    # ----------------------------------------------------------------------
    # No evidence
    # ----------------------------------------------------------------------

    if (
        not chunks
        and not graph_facts
    ):

        return (
            NOT_FOUND_TEXT
            + DISCLAIMER
        )


    # ----------------------------------------------------------------------
    # Context
    # ----------------------------------------------------------------------

    context = (
        build_full_context(
            chunks,
            graph_facts,
        )
    )


    # ----------------------------------------------------------------------
    # Confidence
    # ----------------------------------------------------------------------

    if chunks:

        top_reranker_score = float(
            chunks[0].get(
                "reranker_score",
                0.0,
            )
        )

    else:

        top_reranker_score = 0.0


    print(
        f"[CONFIDENCE] "
        f"top reranker score: "
        f"{top_reranker_score:.4f}"
    )


    # ----------------------------------------------------------------------
    # Step 4 — Scope
    # ----------------------------------------------------------------------

    t0 = time.time()

    in_scope = (
        is_in_scope(
            client,
            query,
            context,
            top_reranker_score,
        )
    )

    print(
        f"[TIMING] is_in_scope: "
        f"{time.time() - t0:.2f}s"
    )


    if not in_scope:

        return (
            OUT_OF_SCOPE_TEXT
            + DISCLAIMER
        )


    # ----------------------------------------------------------------------
    # Step 5 — Draft
    # ----------------------------------------------------------------------

    try:

        t0 = time.time()

        draft = (
            draft_answer(
                client,
                query,
                context,
            )
        )

        print(
            f"[TIMING] draft_answer: "
            f"{time.time() - t0:.2f}s"
        )


        # --------------------------------------------------------------
        # Step 6 — Verify
        # --------------------------------------------------------------

        t0 = time.time()

        verified = (
            verify_answer(
                client,
                query,
                context,
                draft,
            )
        )

        print(
            f"[TIMING] verify_answer: "
            f"{time.time() - t0:.2f}s"
        )


    except GenerationError as exc:

        print(
            f"[DEBUG] GenerationError: "
            f"{exc}"
        )

        return (
            UNVERIFIED_TEXT
            + DISCLAIMER
        )


    # ----------------------------------------------------------------------
    # Citation validation
    # ----------------------------------------------------------------------

    if not has_only_verifiable_citations(
        verified,
        chunks,
        graph_facts,
    ):

        print(
            "[DEBUG] Citation check failed."
        )

        print(
            f"--- draft ---\n{draft}"
        )

        print(
            f"--- verified ---\n{verified}"
        )

        print(
            "--- valid keys ---"
        )

        print(
            _valid_citation_keys(
                chunks,
                graph_facts,
            )
        )

        print(
            "--- citations found ---"
        )

        print(
            _extract_citations(
                verified
            )
        )


        return (
            UNVERIFIED_TEXT
            + DISCLAIMER
        )


    # ----------------------------------------------------------------------
    # Final
    # ----------------------------------------------------------------------

    total_time = (
        time.time()
        - t_start
    )


    print(
        f"[TIMING] TOTAL "
        f"(answer_question): "
        f"{total_time:.2f}s"
    )


    return finalize_answer(
        verified,
        top_reranker_score,
    )


# ==========================================================================
# CLI
# ==========================================================================

def main():

    t_process_start = (
        time.time()
    )


    load_dotenv()


    api_key = os.getenv(
        "GROQ_API_KEY"
    )


    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file."
        )


    api_key = api_key.strip()


    query = (
        " ".join(
            sys.argv[1:]
        )

        or

        "What causes liver cirrhosis?"
    )


    print(
        f"Query: {query}\n"
    )


    client = Groq(
        api_key=api_key
    )


    driver = (
        get_neo4j_driver()
    )


    if driver is None:

        print(
            "(Neo4j unavailable — "
            "answering from vector/BM25 only.)\n"
        )


    try:

        answer = (
            answer_question(
                client,
                driver,
                query,
            )
        )


    finally:

        if driver is not None:

            driver.close()


    print(
        f"[TIMING] TOTAL whole process: "
        f"{time.time() - t_process_start:.2f}s"
    )


    print(
        "=" * 70
    )

    print(
        "ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        answer
    )


# ==========================================================================
# Entry point
# ==========================================================================

if __name__ == "__main__":

    main()
