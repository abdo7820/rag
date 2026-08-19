"""
rag/generate.py

A 6-step grounded-answer chain that turns a question into a grounded,
cited answer, pulling evidence from BOTH the vector/BM25 index and
the Neo4j graph.

Railway / low-RAM mode:
- No torch import
- No sentence-transformers import at module startup
- Embedding + reranking are handled by rag/test_search.py
- When USE_HF_INFERENCE_API=true:
    - Embedding -> Hugging Face
    - Reranking -> Jina API
"""

import os
import pathlib
import re
import sys
import time

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase


# ==========================================================================
# Paths
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
# Retrieval
# ==========================================================================

from test_search import (  # noqa: E402
    semantic_search,
    bm25_search,
    reciprocal_rank_fusion,
    rerank_results,
)


# ==========================================================================
# Configuration
# ==========================================================================

from config import (  # noqa: E402
    GENERATION_MODEL_NAME as MODEL_NAME,
    LOW_CONFIDENCE_THRESHOLD,
    SCOPE_CONFIDENCE_FLOOR,
    GRAPH_FACTS_LIMIT,
    RETRIEVAL_TOP_K,
    GENERATION_MAX_RETRIES,
    GENERATION_RETRY_BACKOFF_S,
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
    "I found some potentially relevant material, but I couldn't "
    "confirm the answer was fully backed by a citation I could "
    "verify, so I'm not showing it rather than risk giving you "
    "an unsupported claim. Please try rephrasing the question "
    "or asking again."
)


# Citation format:
#
# [Source: section, p.7]
# [Graph source: section, p.1]
#
CITATION_RE = re.compile(
    r"\[(Graph source|Source):\s*([^,\]]+?)"
    r",\s*p\.\s*([\w\-]+)\s*\]"
)


class GenerationError(RuntimeError):
    """Raised when a required LLM generation step fails."""


# ==========================================================================
# Medical guardrail
# ==========================================================================

BLOCKED_PATTERNS = re.compile(
    r"\b("
    r"dosage|dose|prescri\w*|"
    r"should i take|"
    r"how much .*(should|do i) take|"
    r"is it safe for me|"
    r"diagnose me|"
    r"do i have"
    r")\b",
    re.IGNORECASE,
)


DISCLAIMER = (
    "\n\n---\n"
    "*This is educational information drawn from a research paper, "
    "not medical advice. Please consult a qualified healthcare "
    "professional for any personal medical decisions.*"
)


# ==========================================================================
# Groq helpers
# ==========================================================================

def _create_completion(
    client: Groq,
    **kwargs,
):
    """
    Creates a Groq completion while handling compatibility
    differences between Groq SDK/model parameter versions.
    """

    try:

        return client.chat.completions.create(
            **kwargs
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


def is_blocked(
    question: str,
) -> bool:

    return bool(
        BLOCKED_PATTERNS.search(
            question
        )
    )


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

            response = _create_completion(
                client,
                **kwargs,
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
                f"{step_name}: model returned "
                "an empty response"
            )

        except Exception as exc:

            last_error = exc

            print(
                f"[DEBUG] {step_name} "
                f"attempt {attempt}/"
                f"{GENERATION_MAX_RETRIES} "
                f"failed: {exc!r}"
            )

        if attempt < GENERATION_MAX_RETRIES:

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
Rewrite the question below into a focused search query for retrieving
passages from a hepatology (liver disease) research paper.

Rules:
- Expand medical abbreviations to their full form alongside the abbreviation.
  Example: HBV -> HBV hepatitis B virus.
- Add closely related medical synonyms if they improve retrieval.
- Keep it short.
- Return a search query, not an answer.
- Do not add commentary.
- Do not answer the question.

Question:
{question}

Search query:
"""


def rewrite_query(
    client: Groq,
    question: str,
) -> str:

    try:

        response = _create_completion(
            client,
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": REWRITE_PROMPT.format(
                        question=question
                    ),
                }
            ],
            temperature=0,
            reasoning_effort="low",
            max_completion_tokens=200,
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

    except Exception:

        return question


# ==========================================================================
# Step 3 — Retrieve
# ==========================================================================

def retrieve_chunks(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:

    semantic_results = (
        semantic_search(query)
    )

    bm25_results = (
        bm25_search(query)
    )

    fused = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
    )

    if not fused:

        return []

    reranked = rerank_results(
        query,
        fused,
    )

    return reranked[:top_k]


def build_chunk_context(
    chunks: list[dict],
) -> str:

    blocks = []

    for c in chunks:

        meta = (
            c.get("metadata", {})
            or {}
        )

        section = meta.get(
            "section",
            "Unknown",
        )

        page = meta.get(
            "page_start",
            "?",
        )

        blocks.append(
            f"[Source: {section}, p.{page}]\n"
            f"{c['text']}"
        )

    return "\n\n".join(
        blocks
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

    try:

        driver = GraphDatabase.driver(
            uri,
            auth=(
                user,
                password,
            ),
        )

        driver.verify_connectivity()

        return driver

    except Exception as exc:

        print(
            "[warn] Neo4j is configured "
            f"but unreachable ({exc}). "
            "Continuing WITHOUT graph facts."
        )

        try:

            driver.close()

        except Exception:

            pass

        return None


# ==========================================================================
# Graph helpers
# ==========================================================================

def _dedupe_graph_facts(
    facts: list[dict],
) -> list[dict]:

    seen = set()
    deduped = []

    for fact in facts:

        key = (
            fact.get("source"),
            fact.get("relation"),
            fact.get("target"),
            fact.get("section"),
            fact.get("page"),
        )

        if key in seen:

            continue

        seen.add(key)

        deduped.append(
            fact
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

        name_lower = (
            name.lower()
        )

        pattern = (
            r"\b"
            + re.escape(name_lower)
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


def _graph_fact_relevance(
    fact: dict,
    question: str,
) -> int:
    """
    Lightweight deterministic relevance scoring.

    This prevents generic graph facts from dominating the context.

    Score:
        +3 source entity mentioned
        +3 target entity mentioned
        +2 relation words overlap
        +1 section contains useful terms
    """

    question_tokens = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question.lower(),
        )
    )

    source = str(
        fact.get(
            "source",
            "",
        )
    ).lower()

    target = str(
        fact.get(
            "target",
            "",
        )
    ).lower()

    relation = str(
        fact.get(
            "relation",
            "",
        )
    ).lower()

    section = str(
        fact.get(
            "section",
            "",
        )
    ).lower()

    score = 0

    if source and source in question.lower():

        score += 3

    if target and target in question.lower():

        score += 3

    relation_tokens = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            relation,
        )
    )

    overlap = (
        question_tokens
        & relation_tokens
    )

    if overlap:

        score += 2

    useful_section_words = {
        "cause",
        "causes",
        "etiology",
        "risk",
        "cirrhosis",
        "liver",
        "disease",
        "pathogenesis",
    }

    if (
        question_tokens
        & set(
            section.split()
        )
        & useful_section_words
    ):

        score += 1

    return score


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
                MATCH (a:Entity)-[r:RELATION]->(b:Entity)
                WHERE a.name IN $names
                   OR b.name IN $names
                RETURN
                    a.name AS source,
                    r.relation AS relation,
                    b.name AS target,
                    r.section AS section,
                    r.page_start AS page
                LIMIT 50
                """,
                names=names,
            )

            facts = [
                dict(record)
                for record in result
            ]

        facts = _dedupe_graph_facts(
            facts
        )

        # --------------------------------------------------------------
        # NEW:
        # Rank graph facts by relevance instead of blindly taking
        # the first N Neo4j relationships.
        # --------------------------------------------------------------

        scored = []

        for fact in facts:

            relevance = (
                _graph_fact_relevance(
                    fact,
                    original_question,
                )
            )

            scored.append(
                (
                    relevance,
                    fact,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # Only keep genuinely relevant graph facts.
        #
        # A score of 0 means the graph relationship is not useful
        # enough for the current question.
        relevant = [
            fact
            for score, fact
            in scored
            if score > 0
        ]

        return relevant[:limit]

    except Exception as exc:

        print(
            "[warn] Neo4j query failed "
            f"mid-session ({exc}). "
            "Answering from vector/BM25 only."
        )

        return []


def build_graph_context(
    facts: list[dict],
) -> str:

    if not facts:

        return ""

    lines = []

    for fact in facts:

        section = (
            fact.get("section")
            or "Unknown"
        )

        page = (
            fact.get("page")
            or "?"
        )

        lines.append(
            f"[Graph source: {section}, p.{page}] "
            f"{fact['source']} "
            f"-[{fact['relation']}]-> "
            f"{fact['target']}"
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
# Step 4 — Scope
# ==========================================================================

SCOPE_PROMPT = """
You will be shown a QUESTION and EVIDENCE excerpts retrieved from a
hepatology/liver disease research paper.

Determine whether:

1. The question is related to liver disease/hepatology.
2. The retrieved evidence is at least topically related.

The evidence does NOT need to fully answer the question.

Respond in exactly two lines:

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

        response = _create_completion(
            client,
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": SCOPE_PROMPT.format(
                        question=question,
                        context=context,
                    ),
                }
            ],
            temperature=0,
            reasoning_effort="low",
            max_completion_tokens=150,
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

    except Exception:

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
# Step 5 — Draft
# ==========================================================================

DRAFT_PROMPT = """
You are a hepatology research assistant answering questions strictly
from the evidence provided below.

There are two evidence types:

1. DOCUMENT EXCERPTS
   Passages from the research paper.

2. KNOWLEDGE GRAPH FACTS
   Relationships extracted from the same research paper.

Rules:

1. Read all evidence before answering.

2. Use ONLY information explicitly supported by the evidence.
   Never use outside knowledge.

3. Prefer DOCUMENT EXCERPTS when they directly answer the question.

4. Use KNOWLEDGE GRAPH FACTS only when they directly support the claim.
   Do NOT use a graph fact merely because an entity happens to appear
   in the question.

5. Every factual claim must have a citation.

6. Document citation format:
   [Source: <section>, p.<page>]

7. Graph citation format:
   [Graph source: <section>, p.<page>]

8. Never fabricate citations.

9. Do not cite INTRODUCTION or REFERENCES merely because they contain
   an entity. Use them only if the actual relationship directly supports
   the answer.

10. If DOCUMENT EXCERPTS directly answer the question, prefer their
    citations over weak/general graph citations.

11. If the evidence only partially answers the question, answer only
    the supported portion.

12. If the evidence does not contain the answer, output EXACTLY:
    "{not_found}"

13. Be concise.

14. Do not add medical disclaimers.
    The application adds them separately.

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

    return _create_completion_with_retry(
        client,
        step_name="draft_answer",
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=2000,
    )


# ==========================================================================
# Step 6a — Verify
# ==========================================================================

VERIFY_PROMPT = """
You are a strict fact-checker.

You are given:
- QUESTION
- EVIDENCE
- DRAFT ANSWER

Check every claim.

Rules:

1. Every factual statement must be supported by the evidence.

2. Every citation must point to evidence that actually supports
   the specific claim.

3. Prefer document citations when a document excerpt directly supports
   the claim.

4. Remove graph citations that are technically valid but do not
   directly support the claim.

5. A graph relationship should only remain if it materially supports
   the factual statement.

6. Every citation must use exactly:

   [Source: <section>, p.<page>]

   OR

   [Graph source: <section>, p.<page>]

7. Do not invent citations.

8. Do not add outside knowledge.

9. If the draft is fully supported, return it unchanged except for
   removing weak/non-supporting graph citations if necessary.

10. If unsupported material exists, remove or correct only that part.

11. If nothing remains supported, output exactly:

{not_found}

Output ONLY the final answer.

QUESTION:
{question}

EVIDENCE:
{context}

DRAFT ANSWER:
{draft}
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

    return _create_completion_with_retry(
        client,
        step_name="verify_answer",
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": VERIFY_PROMPT.format(
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

    for chunk in chunks:

        meta = (
            chunk.get(
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

    for fact in graph_facts:

        section = str(
            fact.get("section")
            or "Unknown"
        ).strip()

        page = str(
            fact.get("page")
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
# Confidence
# ==========================================================================

def _get_retrieval_confidence(
    chunks: list[dict],
) -> float:
    """
    Return a conservative retrieval confidence.

    Jina reranker scores are useful for ranking but should NOT be treated
    as calibrated probabilities.

    We therefore use the top score only as a weak signal and combine it
    with the number of high-quality retrieved results.

    This prevents a good retrieval result from being incorrectly labeled
    low-confidence simply because the raw Jina score is below 0.5.
    """

    if not chunks:

        return 0.0

    scores = []

    for chunk in chunks:

        score = chunk.get(
            "reranker_score"
        )

        if score is None:

            continue

        try:

            scores.append(
                float(score)
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

    if not scores:

        return 0.0

    top_score = max(
        scores
    )

    # Strong evidence:
    #
    # Jina scores above ~0.6 are treated as clearly useful.
    if top_score >= 0.60:

        return 1.0

    # Moderate evidence:
    #
    # Scores in 0.40-0.60 can still be useful.
    if top_score >= 0.40:

        return 0.75

    # Weak but potentially relevant.
    if top_score >= 0.25:

        return 0.50

    return 0.0


def finalize_answer(
    answer: str,
    chunks: list[dict],
) -> str:

    confidence = (
        _get_retrieval_confidence(
            chunks
        )
    )

    confidence_note = ""

    # IMPORTANT:
    # Only show low confidence when there is genuinely weak evidence.
    #
    # This avoids using raw Jina scores as calibrated probabilities.

    if (
        confidence < LOW_CONFIDENCE_THRESHOLD
        and chunks
    ):

        confidence_note = (
            "\n\n⚠️ *Low confidence retrieval — "
            "the available evidence may not be a strong "
            "fit for this question.*"
        )

    return (
        answer
        + confidence_note
        + DISCLAIMER
    )


# ==========================================================================
# Chain orchestration
# ==========================================================================

def answer_question(
    client: Groq,
    driver,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> str:

    if is_blocked(query):

        return (
            "I can't give personal medical advice "
            "(dosage, prescriptions, or diagnosis) — "
            "please talk to a doctor or pharmacist "
            "about that. I can share what the research "
            "paper says about liver disease topics "
            "in general, if that helps."
        )

    t_start = time.time()

    # ----------------------------------------------------------------------
    # Step 1/6 — Query rewrite
    # ----------------------------------------------------------------------

    t0 = time.time()

    search_query = rewrite_query(
        client,
        query,
    )

    print(
        f"[TIMING] rewrite_query: "
        f"{time.time() - t0:.2f}s"
    )

    print(
        f"[DEBUG] Search query: "
        f"{search_query}"
    )

    # ----------------------------------------------------------------------
    # Step 2/6 — Hybrid retrieval
    # ----------------------------------------------------------------------

    t0 = time.time()

    chunks = retrieve_chunks(
        search_query,
        top_k=top_k,
    )

    print(
        f"[TIMING] retrieve_chunks "
        f"(semantic+bm25+rrf+rerank): "
        f"{time.time() - t0:.2f}s"
    )

    # ----------------------------------------------------------------------
    # Step 3/6 — Graph retrieval
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

        for fact in graph_facts:

            section = (
                fact.get(
                    "section"
                )
                or "Unknown"
            )

            page = (
                fact.get(
                    "page"
                )
                or "?"
            )

            print(
                f"{fact['source']} "
                f"-[{fact['relation']}]-> "
                f"{fact['target']} "
                f"(section={section}, p.{page})"
            )

    # ----------------------------------------------------------------------
    # No evidence
    # ----------------------------------------------------------------------

    if not chunks and not graph_facts:

        return (
            NOT_FOUND_TEXT
            + DISCLAIMER
        )

    # ----------------------------------------------------------------------
    # Build context
    # ----------------------------------------------------------------------

    context = build_full_context(
        chunks,
        graph_facts,
    )

    # ----------------------------------------------------------------------
    # Retrieval confidence
    # ----------------------------------------------------------------------

    retrieval_confidence = (
        _get_retrieval_confidence(
            chunks
        )
    )

    print(
        f"[DEBUG] Retrieval confidence: "
        f"{retrieval_confidence:.2f}"
    )

    if chunks:

        scores = [
            c.get(
                "reranker_score"
            )
            for c in chunks
            if c.get(
                "reranker_score"
            ) is not None
        ]

        if scores:

            print(
                f"[DEBUG] Jina scores: "
                f"{[round(float(s), 4) for s in scores]}"
            )

    # ----------------------------------------------------------------------
    # Step 4/6 — Scope
    # ----------------------------------------------------------------------

    t0 = time.time()

    # IMPORTANT:
    #
    # Scope should use the raw top reranker score only for detecting
    # extremely weak retrieval.
    #
    # It should NOT generate a low-confidence warning itself.

    top_reranker_score = (
        float(
            chunks[0].get(
                "reranker_score",
                1.0,
            )
        )
        if chunks
        else 0.0
    )

    in_scope = is_in_scope(
        client,
        query,
        context,
        top_reranker_score,
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
    # Step 5/6 — Draft + verification
    # ----------------------------------------------------------------------

    try:

        t0 = time.time()

        draft = draft_answer(
            client,
            query,
            context,
        )

        print(
            f"[TIMING] draft_answer: "
            f"{time.time() - t0:.2f}s"
        )

        t0 = time.time()

        verified = verify_answer(
            client,
            query,
            context,
            draft,
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
            "--- valid keys ---\n"
            f"{_valid_citation_keys(chunks, graph_facts)}"
        )

        print(
            "--- citations found ---\n"
            f"{_extract_citations(verified)}"
        )

        return (
            UNVERIFIED_TEXT
            + DISCLAIMER
        )

    # ----------------------------------------------------------------------
    # Final cleanup
    # ----------------------------------------------------------------------

    print(
        f"[TIMING] TOTAL "
        f"(answer_question): "
        f"{time.time() - t_start:.2f}s"
    )

    return finalize_answer(
        verified,
        chunks,
    )


# ==========================================================================
# Main
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

    query = (
        " ".join(sys.argv[1:])
        or "What causes liver cirrhosis?"
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

        answer = answer_question(
            client,
            driver,
            query,
        )

    finally:

        if driver is not None:

            driver.close()

    print(
        f"[TIMING] TOTAL "
        f"(whole process, incl. imports): "
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


if __name__ == "__main__":

    main()
