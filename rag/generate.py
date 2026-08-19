"""
rag/generate.py

6-step grounded-answer chain:

1. Query rewrite
2. Hybrid retrieval
3. Neo4j graph retrieval
4. Scope verification
5. Grounded answer generation
6. Answer verification

Retrieval:
    Semantic Search + BM25
            ↓
        RRF Fusion
            ↓
       Jina Reranker

Generation:
    Document Evidence = PRIMARY
    Graph Evidence    = SUPPORTING

Railway:
    USE_HF_INFERENCE_API=true
    HF_API_TOKEN=...
    JINA_API_KEY=...
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

BASE_DIR = (
    pathlib.Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(BASE_DIR),
)

sys.path.insert(
    0,
    str(BASE_DIR / "rag"),
)


# ==========================================================================
# Retrieval
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


# ==========================================================================
# Citation pattern
# ==========================================================================

CITATION_RE = re.compile(
    r"\[(Graph source|Source):\s*([^,\]]+?)"
    r",\s*p\.\s*([\w\-]+)\s*\]"
)


# ==========================================================================
# Generation exception
# ==========================================================================

class GenerationError(RuntimeError):
    """Raised when an LLM generation step fails."""


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
# Groq completion
# ==========================================================================

def _create_completion(
    client: Groq,
    **kwargs,
):

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
# Step 1 — Query Rewrite
# ==========================================================================

REWRITE_PROMPT = """
Rewrite the question below into a focused search query for retrieving
passages from a hepatology/liver disease research paper.

Rules:

- Expand medical abbreviations alongside the abbreviation.
  Example:
  HBV -> HBV hepatitis B virus

- Add closely related medical synonyms when useful.

- Keep the query short.

- Return ONLY the search query.

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

    except Exception as exc:

        print(
            f"[DEBUG] Query rewrite failed: {exc!r}"
        )

        return question


# ==========================================================================
# Step 2 — Hybrid Retrieval
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

    for chunk in chunks:

        metadata = (
            chunk.get(
                "metadata",
                {},
            )
            or {}
        )

        section = metadata.get(
            "section",
            "Unknown",
        )

        page = metadata.get(
            "page_start",
            "?",
        )

        blocks.append(
            f"[Source: {section}, p.{page}]\n"
            f"{chunk.get('text', '')}"
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

    driver = None

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
            "Continuing without graph facts."
        )

        if driver is not None:

            try:
                driver.close()
            except Exception:
                pass

        return None


# ==========================================================================
# Graph entity cache
# ==========================================================================

_all_entity_names = None


def _get_all_entity_names(
    driver,
    database: str,
) -> list[str]:

    global _all_entity_names

    if _all_entity_names is not None:

        return _all_entity_names

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


# ==========================================================================
# Find entities mentioned in question
# ==========================================================================

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
            str(name).lower()
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


# ==========================================================================
# Graph relevance
# ==========================================================================

def _graph_fact_relevance(
    fact: dict,
    question: str,
) -> int:
    """
    Deterministic relevance score for Graph facts.

    Higher score means stronger connection to the question.
    """

    question_lower = (
        question.lower()
    )

    question_tokens = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question_lower,
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

    # --------------------------------------------------------------
    # Entity directly mentioned in question
    # --------------------------------------------------------------

    if source and re.search(
        r"\b"
        + re.escape(source)
        + r"\b",
        question_lower,
    ):

        score += 4

    if target and re.search(
        r"\b"
        + re.escape(target)
        + r"\b",
        question_lower,
    ):

        score += 4

    # --------------------------------------------------------------
    # Relation overlap
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Useful section
    # --------------------------------------------------------------

    useful_section_words = {
        "cause",
        "causes",
        "etiology",
        "risk",
        "cirrhosis",
        "liver",
        "disease",
        "pathogenesis",
        "chronic",
    }

    section_tokens = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            section,
        )
    )

    if (
        question_tokens
        & section_tokens
        & useful_section_words
    ):

        score += 1

    # --------------------------------------------------------------
    # Penalize generic reference sections
    # --------------------------------------------------------------

    if "reference" in section:

        score -= 5

    if section.strip() == "introduction":

        score -= 2

    return score


# ==========================================================================
# Deduplicate Graph facts
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

        seen.add(
            key
        )

        deduped.append(
            fact
        )

    return deduped


# ==========================================================================
# Graph retrieval
# ==========================================================================

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

        # Only keep actually relevant facts.
        relevant = [
            fact
            for score, fact in scored
            if score > 0
        ]

        return relevant[:limit]

    except Exception as exc:

        print(
            "[warn] Neo4j query failed: "
            f"{exc}. "
            "Continuing without graph facts."
        )

        return []


# ==========================================================================
# Graph context
# ==========================================================================

def build_graph_context(
    facts: list[dict],
) -> str:

    if not facts:

        return ""

    lines = []

    for fact in facts:

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

        lines.append(
            f"[Graph source: {section}, p.{page}] "
            f"{fact.get('source', 'Unknown')} "
            f"-[{fact.get('relation', 'RELATED_TO')}]-> "
            f"{fact.get('target', 'Unknown')}"
        )

    return "\n".join(
        lines
    )


# ==========================================================================
# Full context
# ==========================================================================

def build_full_context(
    chunks: list[dict],
    graph_facts: list[dict],
) -> str:

    parts = []

    document_context = (
        build_chunk_context(
            chunks
        )
    )

    if document_context:

        parts.append(
            "### DOCUMENT EXCERPTS\n\n"
            + document_context
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

    if not parts:

        return "(no evidence retrieved)"

    return "\n\n".join(
        parts
    )


# ==========================================================================
# Step 3 — Scope
# ==========================================================================

SCOPE_PROMPT = """
You will be shown a QUESTION and EVIDENCE retrieved from a
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

    except Exception as exc:

        print(
            f"[DEBUG] Scope classification failed: "
            f"{exc!r}"
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
# Step 4 — Draft
# ==========================================================================

DRAFT_PROMPT = """
You are a hepatology research assistant answering questions STRICTLY
from the evidence provided below.

There are TWO evidence types:

1. DOCUMENT EXCERPTS
   Direct passages retrieved from the original research paper.

2. KNOWLEDGE GRAPH FACTS
   Relationships extracted from the same research paper.

============================================================
EVIDENCE PRIORITY
============================================================

Use evidence in this priority order:

1. DOCUMENT EXCERPTS = PRIMARY EVIDENCE
2. KNOWLEDGE GRAPH FACTS = SUPPORTING EVIDENCE

If a DOCUMENT EXCERPT directly answers the question, it MUST be the
primary basis of the answer.

Use GRAPH FACTS only to:

- support a claim already supported by the document, OR
- add a directly relevant relationship that is not explicitly stated
  in the retrieved document.

Never allow a graph fact to replace strong document evidence.

============================================================
ANSWER COMPLETENESS
============================================================

When the document evidence explicitly lists multiple major causes,
risk factors, mechanisms, or categories relevant to the question,
include ALL directly supported major items.

For example, if the evidence explicitly identifies:

- MASLD
- HBV
- HCV
- ALD

as causes of cirrhosis, do NOT answer with only HBV and HCV.

Do not omit an important directly supported item simply because a graph
fact mentions another cause.

============================================================
GROUNDING RULES
============================================================

1. Use ONLY information explicitly supported by the evidence.

2. Never use outside medical knowledge.

3. Every factual claim must have a citation.

4. Document citation format:

   [Source: <section>, p.<page>]

5. Graph citation format:

   [Graph source: <section>, p.<page>]

6. Never fabricate citations.

7. Never cite a graph fact merely because an entity appears in the
   question.

8. A graph citation is valid only when the specific relationship
   directly supports the claim.

9. Prefer ONE strong document citation when one document excerpt
   supports several related claims.

10. Avoid unnecessary graph citations when the document already
    provides sufficient evidence.

11. Do not cite INTRODUCTION or REFERENCES merely because they contain
    an entity.

12. If the evidence only partially answers the question, answer only
    the supported portion — a partial, ranged, or regionally-varying
    figure (e.g. "38.0% globally between 2016-2019, with regional
    variation from X% to Y%") STILL counts as answering the question.
    Do not treat "the evidence gives a range/trend instead of one
    single fixed number" as a reason to refuse.

13. Only output the exact refusal below if the evidence is genuinely
    unrelated to the question or contains nothing that addresses it
    at all:

I don't know based on the available sources.

    Do NOT use this refusal just because the answer is a range, an
    estimate, drawn from multiple sub-populations, or otherwise not a
    single clean number. Give the best supported answer instead.

14. Be concise but complete.

15. Do not add medical disclaimers.
    The application adds them separately.

============================================================
QUESTION
============================================================

{question}

============================================================
EVIDENCE
============================================================

{context}

============================================================
FINAL ANSWER
============================================================
"""


def draft_answer(
    client: Groq,
    query: str,
    context: str,
) -> str:

    prompt = DRAFT_PROMPT.format(
        question=query,
        context=context,
    )

    return _create_completion_with_retry(
        client,
        step_name="draft_answer",
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0,
        reasoning_effort="medium",
        max_completion_tokens=2000,
    )


# ==========================================================================
# Step 5 — Verification
# ==========================================================================

VERIFY_PROMPT = """
You are a strict fact-checker for a hepatology research QA system.

You are given:

1. QUESTION
2. DOCUMENT EXCERPTS
3. KNOWLEDGE GRAPH FACTS
4. DRAFT ANSWER

Your job is to produce the FINAL grounded answer.

============================================================
EVIDENCE PRIORITY
============================================================

DOCUMENT EXCERPTS are PRIMARY evidence.

KNOWLEDGE GRAPH FACTS are SECONDARY supporting evidence.

If the document directly supports a claim, keep the document evidence
as the primary citation.

Do NOT replace strong document evidence with a graph citation.

============================================================
COMPLETENESS CHECK
============================================================

Before producing the final answer, check whether the DOCUMENT EXCERPTS
explicitly list multiple causes, risk factors, mechanisms, or categories.

If several major items are directly supported and relevant to the
question, retain ALL of them.

For example, if the evidence explicitly supports:

- MASLD
- HBV
- HCV
- ALD

then the final answer should mention all four when answering a question
about causes of cirrhosis.

Do not accidentally reduce a multi-item answer to only the entities
that happen to appear in the graph.

============================================================
CITATION RULES
============================================================

1. Every factual claim must be supported by the evidence.

2. Document citation:

   [Source: <section>, p.<page>]

3. Graph citation:

   [Graph source: <section>, p.<page>]

4. Every citation must correspond to actual evidence provided.

5. Remove fabricated citations.

6. Remove graph citations that do not directly support the claim.

7. Do not cite INTRODUCTION or REFERENCES merely because an entity
   appears there.

8. If a document excerpt directly supports a claim, prefer its citation.

9. A graph relationship should only remain if it materially supports
   the exact claim.

10. Do not add outside medical knowledge.

============================================================
STYLE
============================================================

- Answer the question directly.
- Keep the answer concise.
- Preserve important directly supported details.
- Do not add unsupported explanations.
- Do not add medical advice.
- Do not mention this verification process.

If the draft is fully supported, return the corrected final answer.

If unsupported material exists, remove or correct only that material.

A partial, ranged, or regionally-varying figure that IS supported by
the evidence (e.g. "38.0% globally between 2016-2019, ranging
regionally from X% to Y%") counts as a valid, complete answer. Do not
discard a supported draft just because it reports a range or trend
instead of one single fixed number.

Only if NOTHING in the draft remains supported by the evidence after
removing unsupported claims, output exactly:

I don't know based on the available sources.

============================================================
QUESTION
============================================================

{question}

============================================================
DOCUMENT EXCERPTS
============================================================

{document_context}

============================================================
KNOWLEDGE GRAPH FACTS
============================================================

{graph_context}

============================================================
DRAFT ANSWER
============================================================

{draft}

============================================================
FINAL ANSWER
============================================================
"""


def verify_answer(
    client: Groq,
    query: str,
    chunks: list[dict],
    graph_facts: list[dict],
    draft: str,
) -> str:

    document_context = (
        build_chunk_context(
            chunks
        )
    )

    graph_context = (
        build_graph_context(
            graph_facts
        )
    )

    prompt = VERIFY_PROMPT.format(
        question=query,
        document_context=(
            document_context
            or "(none)"
        ),
        graph_context=(
            graph_context
            or "(none)"
        ),
        draft=draft,
    )

    return _create_completion_with_retry(
        client,
        step_name="verify_answer",
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        reasoning_effort="medium",
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


def _normalize_citation_key(
    kind: str,
    section: str,
    page: str,
) -> tuple[str, str, str]:
    """
    Normalize a citation key so that harmless differences in how the
    LLM re-types a section name or page number (case, leading numbering
    like "1. ", extra whitespace, stray punctuation) don't cause a
    valid, well-grounded citation to be rejected.
    """

    kind_norm = (
        kind.strip().lower()
    )

    section_norm = section.strip().lower()

    # Strip leading numbering like "1. ", "2) ", "3 - " etc.
    section_norm = re.sub(
        r"^\d+[\.\)\-]\s*",
        "",
        section_norm,
    )

    # Collapse internal whitespace.
    section_norm = re.sub(
        r"\s+",
        " ",
        section_norm,
    ).strip()

    # Keep only alphanumerics/hyphens for page comparison
    # (handles "p.12", "12 ", "12-13", etc. consistently).
    page_norm = re.sub(
        r"[^\w\-]",
        "",
        page.strip().lower(),
    )

    return (
        kind_norm,
        section_norm,
        page_norm,
    )


def _valid_citation_keys(
    chunks: list[dict],
    graph_facts: list[dict],
) -> set[tuple[str, str, str]]:

    valid = set()

    for chunk in chunks:

        metadata = (
            chunk.get(
                "metadata",
                {},
            )
            or {}
        )

        section = str(
            metadata.get(
                "section",
                "Unknown",
            )
        ).strip()

        page = str(
            metadata.get(
                "page_start",
                "?",
            )
        ).strip()

        valid.add(
            _normalize_citation_key(
                "Source",
                section,
                page,
            )
        )

    for fact in graph_facts:

        section = str(
            fact.get(
                "section"
            )
            or "Unknown"
        ).strip()

        page = str(
            fact.get(
                "page"
            )
            or "?"
        ).strip()

        valid.add(
            _normalize_citation_key(
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
        _normalize_citation_key(*citation)
        in valid_keys
        for citation in citations
    )


# ==========================================================================
# Confidence
# ==========================================================================

def _get_retrieval_confidence(
    chunks: list[dict],
) -> float:
    """
    Convert raw Jina reranker scores into a coarse confidence signal.

    Jina scores are ranking scores, not calibrated probabilities.
    Therefore they should not be interpreted directly as probability.
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

    if top_score >= 0.60:

        return 1.0

    if top_score >= 0.40:

        return 0.75

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

    if (
        confidence
        < LOW_CONFIDENCE_THRESHOLD
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
# Main answer chain
# ==========================================================================

def answer_question(
    client: Groq,
    driver,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> str:

    # ----------------------------------------------------------------------
    # Medical guardrail
    # ----------------------------------------------------------------------

    if is_blocked(query):

        return (
            "I can't give personal medical advice "
            "(dosage, prescriptions, or diagnosis). "
            "I can share what the research paper says "
            "about liver disease topics in general."
        )

    t_start = time.time()

    # ----------------------------------------------------------------------
    # Step 1 — Rewrite
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
    # Step 2 — Hybrid retrieval + Reranking
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
    # Step 3 — Graph retrieval
    # ----------------------------------------------------------------------

    t0 = time.time()

    graph_facts = (
        retrieve_graph_facts(
            driver,
            query,
            limit=GRAPH_FACTS_LIMIT,
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
                f"{fact.get('source')} "
                f"-[{fact.get('relation')}]-> "
                f"{fact.get('target')} "
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
    # Confidence
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

    scores = [
        chunk.get(
            "reranker_score"
        )
        for chunk in chunks
        if chunk.get(
            "reranker_score"
        ) is not None
    ]

    if scores:

        print(
            "[DEBUG] Jina scores: "
            f"{[round(float(s), 4) for s in scores]}"
        )

    # ----------------------------------------------------------------------
    # Step 4 — Scope
    # ----------------------------------------------------------------------

    t0 = time.time()

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
    # Step 5 — Draft
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

        print(
            "[DEBUG] Draft answer:\n"
            f"{draft}"
        )

        # --------------------------------------------------------------
        # Step 6 — Verification
        # --------------------------------------------------------------

        t0 = time.time()

        verified = verify_answer(
            client,
            query,
            chunks,
            graph_facts,
            draft,
        )

        print(
            f"[TIMING] verify_answer: "
            f"{time.time() - t0:.2f}s"
        )

        print(
            "[DEBUG] Verified answer:\n"
            f"{verified}"
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
            "[DEBUG] Citation validation failed."
        )

        print(
            "--- DRAFT ---"
        )

        print(
            draft
        )

        print(
            "--- VERIFIED ---"
        )

        print(
            verified
        )

        print(
            "--- VALID CITATIONS (normalized) ---"
        )

        print(
            _valid_citation_keys(
                chunks,
                graph_facts,
            )
        )

        print(
            "--- FOUND CITATIONS (raw) ---"
        )

        print(
            _extract_citations(
                verified
            )
        )

        print(
            "--- FOUND CITATIONS (normalized) ---"
        )

        print(
            [
                _normalize_citation_key(*c)
                for c in _extract_citations(verified)
            ]
        )

        return (
            UNVERIFIED_TEXT
            + DISCLAIMER
        )

    # ----------------------------------------------------------------------
    # Final answer
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

            try:
                driver.close()
            except Exception:
                pass

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
