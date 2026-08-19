"""
rag/generate.py

6-step grounded-answer chain for the Liver RAG system.

Pipeline:
    1. Guardrail
    2. Query rewrite
    3. Hybrid retrieval:
       Chroma semantic + BM25 -> RRF -> reranking
    4. Scope check
    5. Grounded draft answer
    6. Verification + citation validation + finalization

The retrieval/reranking implementation lives in test_search.py.

For Railway / low-RAM deployment:
    - No torch import here.
    - No local embedding/reranker models are loaded here.
    - test_search.py handles HF API mode when:
          USE_HF_INFERENCE_API=true
"""

import os
import pathlib
import re
import sys
import time

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase


# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR / "rag"))
sys.path.insert(0, str(BASE_DIR))


# --------------------------------------------------------------------------
# Retrieval imports
# --------------------------------------------------------------------------

from test_search import (  # noqa: E402
    semantic_search,
    bm25_search,
    reciprocal_rank_fusion,
    rerank_results,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

from config import (  # noqa: E402
    GENERATION_MODEL_NAME as MODEL_NAME,
    LOW_CONFIDENCE_THRESHOLD,
    SCOPE_CONFIDENCE_FLOOR,
    GRAPH_FACTS_LIMIT,
    RETRIEVAL_TOP_K,
    GENERATION_MAX_RETRIES,
    GENERATION_RETRY_BACKOFF_S,
)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

NOT_FOUND_TEXT = "I don't know based on the available sources."

OUT_OF_SCOPE_TEXT = (
    "That's outside what this hepatology paper covers, so I can't answer it "
    "from these sources."
)

UNVERIFIED_TEXT = (
    "I found some potentially relevant material, but I couldn't confirm the "
    "answer was fully backed by a citation I could verify, so I'm not "
    "showing it rather than risk giving you an unsupported claim. Please "
    "try rephrasing the question or asking again."
)

DISCLAIMER = (
    "\n\n---\n"
    "*This is educational information drawn from a research paper, not "
    "medical advice. Please consult a qualified healthcare professional "
    "for any personal medical decisions.*"
)


# --------------------------------------------------------------------------
# Citation parser
# --------------------------------------------------------------------------

CITATION_RE = re.compile(
    r"\[(Graph source|Source):\s*([^,\]]+?)\s*,\s*p\.\s*([\w\-]+)\s*\]"
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class GenerationError(RuntimeError):
    """
    Raised when a required LLM generation step fails after retries.
    """

    pass


# --------------------------------------------------------------------------
# Step 1 — Guardrail
# --------------------------------------------------------------------------

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


def is_blocked(question: str) -> bool:
    return bool(BLOCKED_PATTERNS.search(question))


# --------------------------------------------------------------------------
# Groq compatibility wrapper
# --------------------------------------------------------------------------

def _create_completion(client: Groq, **kwargs):
    """
    Supports both newer and older Groq SDK versions.

    New SDK:
        reasoning_effort
        max_completion_tokens

    Older SDK:
        max_tokens
    """

    try:
        return client.chat.completions.create(**kwargs)

    except TypeError as exc:

        if "reasoning_effort" in str(exc):
            kwargs.pop("reasoning_effort", None)
            return _create_completion(client, **kwargs)

        if "max_completion_tokens" in str(exc):
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            return _create_completion(client, **kwargs)

        raise


# --------------------------------------------------------------------------
# Generic retry wrapper for generation
# --------------------------------------------------------------------------

def _create_completion_with_retry(
    client: Groq,
    *,
    step_name: str,
    **kwargs,
) -> str:
    """
    Retry important generation steps.

    Empty model responses are treated as failures.
    """

    last_error = None

    for attempt in range(1, GENERATION_MAX_RETRIES + 1):

        try:

            response = _create_completion(
                client,
                **kwargs,
            )

            text = (
                response.choices[0].message.content or ""
            ).strip()

            if text:
                return text

            last_error = RuntimeError(
                f"{step_name}: model returned an empty response"
            )

        except Exception as exc:
            last_error = exc

            print(
                f"[DEBUG] {step_name} "
                f"attempt {attempt}/{GENERATION_MAX_RETRIES} "
                f"failed: {exc!r}"
            )

        if attempt < GENERATION_MAX_RETRIES:
            time.sleep(
                GENERATION_RETRY_BACKOFF_S * attempt
            )

    raise GenerationError(
        f"{step_name} failed after "
        f"{GENERATION_MAX_RETRIES} attempts: "
        f"{last_error}"
    ) from last_error


# ==========================================================================
# STEP 2 — QUERY REWRITE
# ==========================================================================

REWRITE_PROMPT = """
Rewrite the question below into a focused search query
for retrieving passages from a hepatology (liver disease)
research paper.

Rules:

- Expand medical abbreviations to their full form alongside
  the abbreviation.
  Example:
  HBV -> HBV hepatitis B virus

- Add closely related medical synonyms if they improve retrieval.

- Keep it short.

- Output a search query, not an answer.

- Do not answer the question.

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
            response.choices[0].message.content or ""
        ).strip()

        return rewritten if rewritten else question

    except Exception as exc:

        print(
            f"[DEBUG] Query rewrite failed: {exc!r}"
        )

        # Rewrite is an enhancement only.
        # Never let it break retrieval.
        return question


# ==========================================================================
# STEP 3 — RETRIEVAL
# ==========================================================================

def retrieve_chunks(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Hybrid retrieval:

        semantic
             +
        BM25
             ↓
        RRF fusion
             ↓
        reranker
             ↓
        top_k
    """

    semantic_results = semantic_search(query)

    bm25_results = bm25_search(query)

    fused = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
    )

    reranked = rerank_results(
        query,
        fused,
    )

    return reranked[:top_k]


# ==========================================================================
# DOCUMENT CONTEXT
# ==========================================================================

def build_chunk_context(
    chunks: list[dict],
) -> str:

    blocks = []

    for c in chunks:

        meta = c.get("metadata", {})

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

    return "\n\n".join(blocks)


# ==========================================================================
# NEO4J
# ==========================================================================

def get_neo4j_driver():

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([
        uri,
        user,
        password,
    ]):
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
            "[warn] Neo4j is configured but unreachable: "
            f"{exc}"
        )

        print(
            "[warn] Continuing WITHOUT graph facts."
        )

        driver.close()

        return None

    return driver


# ==========================================================================
# GRAPH DEDUPLICATION
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

        seen.add(key)
        deduped.append(f)

    return deduped


# ==========================================================================
# GRAPH RETRIEVAL
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

    question_lower = original_question.lower()

    try:

        with driver.session(
            database=database
        ) as session:

            candidate_names = session.run(
                """
                MATCH (e:Entity)
                WHERE size(e.name) > 2
                  AND toLower($question)
                      CONTAINS toLower(e.name)
                RETURN DISTINCT e.name AS name
                """,
                question=question_lower,
            )

            names = [
                r["name"]
                for r in candidate_names
            ]

            if not names:
                return []

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
            "[warn] Neo4j query failed: "
            f"{exc}"
        )

        print(
            "[warn] Answering from vector/BM25 "
            "only for this question."
        )

        return []


# ==========================================================================
# GRAPH CONTEXT
# ==========================================================================

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
            f"[Graph source: {section}, p.{page}] "
            f"{f['source']} "
            f"-[{f['relation']}]-> "
            f"{f['target']}"
        )

    return "\n".join(lines)


# ==========================================================================
# FULL CONTEXT
# ==========================================================================

def build_full_context(
    chunks: list[dict],
    graph_facts: list[dict],
) -> str:

    parts = []

    chunk_context = build_chunk_context(
        chunks
    )

    if chunk_context:

        parts.append(
            "### DOCUMENT EXCERPTS\n\n"
            + chunk_context
        )

    graph_context = build_graph_context(
        graph_facts
    )

    if graph_context:

        parts.append(
            "### KNOWLEDGE GRAPH FACTS\n\n"
            + graph_context
        )

    if not parts:
        return "(no evidence retrieved)"

    return "\n\n".join(parts)


# ==========================================================================
# STEP 4 — SCOPE CHECK
# ==========================================================================

SCOPE_PROMPT = """
You will be shown a QUESTION and EVIDENCE excerpts
retrieved from a hepatology research paper.

Determine:

1. Whether the question is related to hepatology,
   liver disease, or liver science.

2. Whether the evidence is at least topically related.

Important:
The evidence does NOT need to completely answer the
question for the question to be considered in scope.

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
            response.choices[0].message.content
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

        return "YES" in verdict_line

    except Exception as exc:

        print(
            f"[DEBUG] Scope classifier failed: {exc!r}"
        )

        # Don't block if the LLM classifier itself fails.
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

    # Strong retrieval means the question is already
    # sufficiently supported by retrieval.
    if not retrieval_is_weak:
        return True

    llm_says_in_scope = classify_scope(
        client,
        question,
        context,
    )

    return llm_says_in_scope


# ==========================================================================
# STEP 5 — DRAFT ANSWER
# ==========================================================================

DRAFT_PROMPT = """
You are a hepatology research assistant.

Answer the user's question STRICTLY from the evidence
provided below.

There are two types of evidence:

1. DOCUMENT EXCERPTS

Each document excerpt has:

[Source: <section>, p.<page>]

2. KNOWLEDGE GRAPH FACTS

Each graph fact has:

[Graph source: <section>, p.<page>]

Rules:

1. Read ALL evidence before answering.

2. Only state facts directly supported by the evidence.

3. Never use outside knowledge.

4. Every factual claim MUST have a citation.

5. Use exactly:

[Source: <section>, p.<page>]

or:

[Graph source: <section>, p.<page>]

6. Never invent:
   - citations
   - pages
   - statistics
   - relationships

7. If the evidence only partially answers the question,
   provide only the supported information.

8. If the evidence does NOT contain the answer,
   respond EXACTLY:

I don't know based on the available sources.

9. Prefer combining document evidence and graph evidence
   when both support the answer.

10. Be concise.

11. Do not add medical disclaimers.
    They are added separately.

EVIDENCE:
{context}
"""


def draft_answer(
    client: Groq,
    query: str,
    context: str,
) -> str:

    system_prompt = DRAFT_PROMPT.format(
        context=context
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
# STEP 6A — VERIFY
# ==========================================================================

VERIFY_PROMPT = """
You are a strict fact-checker.

You are given:

QUESTION
EVIDENCE
DRAFT ANSWER

Check the draft claim by claim.

Rules:

1. Every factual statement must be supported by
   something literally present in the evidence.

2. Every citation must correspond to a real source
   in the evidence.

3. Every citation must use exactly:

[Source: <section>, p.<page>]

or:

[Graph source: <section>, p.<page>]

4. Section AND page are mandatory.

5. If a citation is wrong, fix it using the actual
   evidence.

6. If a claim is unsupported, remove or correct it.

7. If removing unsupported content leaves nothing,
   output exactly:

I don't know based on the available sources.

8. If the draft is already fully supported,
   output it unchanged.

9. Output ONLY the final answer.
   No explanation.
   No preamble.

QUESTION:
{question}

EVIDENCE:
{context}

DRAFT ANSWER:
{draft}
"""


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
# CITATION VALIDATION
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
        in CITATION_RE.findall(text)
    ]


def _valid_citation_keys(
    chunks: list[dict],
    graph_facts: list[dict],
) -> set[tuple[str, str, str]]:

    valid = set()

    for c in chunks:

        meta = c.get(
            "metadata",
            {},
        ) or {}

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
            f.get("section")
            or "Unknown"
        ).strip()

        page = str(
            f.get("page")
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

    citations = _extract_citations(
        answer
    )

    # NOT_FOUND / OUT_OF_SCOPE are allowed.
    if not citations:
        return True

    valid_keys = _valid_citation_keys(
        chunks,
        graph_facts,
    )

    return all(
        citation in valid_keys
        for citation in citations
    )


# ==========================================================================
# FINALIZE
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
# MAIN 6-STEP CHAIN
# ==========================================================================

def answer_question(
    client: Groq,
    driver,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> str:

    # ------------------------------------------------------
    # STEP 1 — Guardrail
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # STEP 2 — Query rewrite
    # ------------------------------------------------------

    t0 = time.time()

    search_query = rewrite_query(
        client,
        query,
    )

    print(
        f"[DEBUG] Original query: {query}"
    )

    print(
        f"[DEBUG] Search query:   {search_query}"
    )

    print(
        f"[TIMING] rewrite_query: "
        f"{time.time() - t0:.2f}s"
    )

    # ------------------------------------------------------
    # STEP 3 — Hybrid retrieval
    # ------------------------------------------------------

    t0 = time.time()

    chunks = retrieve_chunks(
        search_query,
        top_k=top_k,
    )

    print(
        "[TIMING] retrieve_chunks "
        "(semantic+bm25+rrf+rerank): "
        f"{time.time() - t0:.2f}s"
    )

    # Show retrieved chunks for debugging
    print("\n=== Retrieved chunks ===")

    if not chunks:

        print("(none)")

    else:

        for i, c in enumerate(
            chunks,
            start=1,
        ):

            meta = c.get(
                "metadata",
                {},
            ) or {}

            print(
                f"\n--- Result {i} ---"
            )

            print(
                "chunk_id:",
                meta.get(
                    "chunk_id",
                    c.get("chunk_id"),
                ),
            )

            print(
                "section:",
                meta.get(
                    "section",
                    "Unknown",
                ),
            )

            print(
                "page:",
                meta.get(
                    "page_start",
                    "?",
                ),
            )

            print(
                "reranker_score:",
                c.get(
                    "reranker_score",
                    "N/A",
                ),
            )

            print(
                "text:",
                c.get(
                    "text",
                    "",
                )[:800],
            )

    # ------------------------------------------------------
    # Graph retrieval
    # ------------------------------------------------------

    t0 = time.time()

    graph_facts = retrieve_graph_facts(
        driver,
        query,
    )

    print(
        f"[TIMING] retrieve_graph_facts: "
        f"{time.time() - t0:.2f}s"
    )

    print("\n=== Graph facts ===")

    if not graph_facts:

        print(
            "(none retrieved — either no matching "
            "entities or Neo4j unavailable)"
        )

    else:

        for f in graph_facts:

            section = (
                f.get("section")
                or "Unknown"
            )

            page = (
                f.get("page")
                or "?"
            )

            print(
                f"{f['source']} "
                f"-[{f['relation']}]-> "
                f"{f['target']} "
                f"(section={section}, p.{page})"
            )

    # ------------------------------------------------------
    # No evidence
    # ------------------------------------------------------

    if not chunks and not graph_facts:

        return (
            NOT_FOUND_TEXT
            + DISCLAIMER
        )

    # ------------------------------------------------------
    # Build context
    # ------------------------------------------------------

    context = build_full_context(
        chunks,
        graph_facts,
    )

    # Important:
    # reranker score is produced by test_search.py.
    top_reranker_score = (
        chunks[0].get(
            "reranker_score",
            1.0,
        )
        if chunks
        else 0.0
    )

    print(
        "\n[DEBUG] Top reranker score:",
        top_reranker_score,
    )

    # ------------------------------------------------------
    # STEP 4 — Scope
    # ------------------------------------------------------

    t0 = time.time()

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

    # ------------------------------------------------------
    # STEP 5 — Draft
    # ------------------------------------------------------

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
            "\n=== Draft answer ==="
        )

        print(draft)

        # --------------------------------------------------
        # STEP 6A — Verify
        # --------------------------------------------------

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

        print(
            "\n=== Verified answer ==="
        )

        print(verified)

    except GenerationError as exc:

        print(
            f"[DEBUG] GenerationError: {exc}"
        )

        return (
            UNVERIFIED_TEXT
            + DISCLAIMER
        )

    # ------------------------------------------------------
    # Citation validation
    # ------------------------------------------------------

    if not has_only_verifiable_citations(
        verified,
        chunks,
        graph_facts,
    ):

        print(
            "\n[DEBUG] Citation validation failed."
        )

        print(
            "--- Draft ---"
        )

        print(draft)

        print(
            "--- Verified ---"
        )

        print(verified)

        print(
            "--- Valid citation keys ---"
        )

        print(
            _valid_citation_keys(
                chunks,
                graph_facts,
            )
        )

        print(
            "--- Citations found ---"
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

    # ------------------------------------------------------
    # STEP 6B — Finalize
    # ------------------------------------------------------

    print(
        f"[TIMING] TOTAL "
        f"(answer_question): "
        f"{time.time() - t_start:.2f}s"
    )

    return finalize_answer(
        verified,
        top_reranker_score,
    )


# ==========================================================================
# CLI
# ==========================================================================

def main():

    t_process_start = time.time()

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
        or
        "What causes liver cirrhosis?"
    )

    print(
        f"Query: {query}\n"
    )

    client = Groq(
        api_key=api_key
    )

    driver = get_neo4j_driver()

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
        f"(whole process): "
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

    print(answer)


if __name__ == "__main__":
    main()
