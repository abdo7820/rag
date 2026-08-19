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

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "rag"))


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

NOT_FOUND_TEXT = "I don't know based on the available sources."

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

CITATION_RE = re.compile(
    r"\[(Graph source|Source):\s*([^,\]]+?),\s*p\.\s*([\w\-]+)\s*\]"
)


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


def _create_completion(client: Groq, **kwargs):
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


def is_blocked(question: str) -> bool:
    return bool(BLOCKED_PATTERNS.search(question))


def _create_completion_with_retry(client: Groq, *, step_name: str, **kwargs) -> str:
    last_error = None

    for attempt in range(1, GENERATION_MAX_RETRIES + 1):
        try:
            response = _create_completion(client, **kwargs)
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
            last_error = RuntimeError(f"{step_name}: model returned an empty response")
        except Exception as exc:
            last_error = exc
            print(f"[DEBUG] {step_name} attempt {attempt}/{GENERATION_MAX_RETRIES} failed: {exc!r}")

        if attempt < GENERATION_MAX_RETRIES:
            time.sleep(GENERATION_RETRY_BACKOFF_S * attempt)

    raise GenerationError(
        f"{step_name} failed after {GENERATION_MAX_RETRIES} attempts: {last_error}"
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


def rewrite_query(client: Groq, question: str) -> str:
    try:
        response = _create_completion(
            client,
            model=MODEL_NAME,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
            temperature=0,
            reasoning_effort="low",
            max_completion_tokens=200,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten if rewritten else question
    except Exception as exc:
        print(f"[DEBUG] Query rewrite failed: {exc!r}")
        return question


# ==========================================================================
# Step 2 — Hybrid Retrieval
# ==========================================================================

def retrieve_chunks(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    semantic_results = semantic_search(query)
    bm25_results = bm25_search(query)
    fused = reciprocal_rank_fusion(semantic_results, bm25_results)

    if not fused:
        return []

    reranked = rerank_results(query, fused)
    return reranked[:top_k]


def build_chunk_context(chunks: list[dict]) -> str:
    blocks = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}
        section = metadata.get("section", "Unknown")
        page = metadata.get("page_start", "?")
        blocks.append(f"[Source: {section}, p.{page}]\n{chunk.get('text', '')}")

    return "\n\n".join(blocks)


# ==========================================================================
# Neo4j
# ==========================================================================

def get_neo4j_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        return None

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        driver.verify_connectivity()
        return driver
    except Exception as exc:
        print(f"[warn] Neo4j is configured but unreachable ({exc}). "
              "Continuing without graph facts.")
        driver.close()
        return None


# Cached once per process — the entity vocabulary doesn't change at runtime.
_all_entity_names = None


def _get_all_entity_names(driver, database: str) -> list[str]:
    global _all_entity_names

    if _all_entity_names is None:
        with driver.session(database=database) as session:
            result = session.run(
                "MATCH (e:Entity) WHERE size(e.name) > 2 RETURN DISTINCT e.name AS name"
            )
            _all_entity_names = [record["name"] for record in result]

    return _all_entity_names


def _find_mentioned_entities(question: str, all_names: list[str]) -> list[str]:
    question_lower = question.lower()
    matched = []

    for name in all_names:
        pattern = r"\b" + re.escape(str(name).lower()) + r"\b"
        if re.search(pattern, question_lower):
            matched.append(name)

    return matched


def _dedupe_graph_facts(facts: list[dict]) -> list[dict]:
    seen = set()
    deduped = []

    for fact in facts:
        key = (fact.get("source"), fact.get("relation"), fact.get("target"), fact.get("section"), fact.get("page"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)

    return deduped


def retrieve_graph_facts(driver, original_question: str, limit: int = GRAPH_FACTS_LIMIT) -> list[dict]:
    """
    Uses the ORIGINAL question (not the rewritten query) so entity names
    still match the user's own wording, e.g. "HBV" not "HBV hepatitis B virus".
    """

    if driver is None:
        return []

    database = os.getenv("NEO4J_DATABASE", "neo4j")

    try:
        all_names = _get_all_entity_names(driver, database)
        names = _find_mentioned_entities(original_question, all_names)

        if not names:
            return []

        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (a:Entity)-[r:RELATION]->(b:Entity)
                WHERE a.name IN $names OR b.name IN $names
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
            facts = [dict(record) for record in result]

        return _dedupe_graph_facts(facts)

    except Exception as exc:
        print(f"[warn] Neo4j query failed: {exc}. Continuing without graph facts.")
        return []


def build_graph_context(facts: list[dict]) -> str:
    if not facts:
        return ""

    lines = []

    for fact in facts:
        section = fact.get("section") or "Unknown"
        page = fact.get("page") or "?"
        lines.append(
            f"[Graph source: {section}, p.{page}] "
            f"{fact.get('source', 'Unknown')} -[{fact.get('relation', 'RELATED_TO')}]-> "
            f"{fact.get('target', 'Unknown')}"
        )

    return "\n".join(lines)


def build_full_context(chunks: list[dict], graph_facts: list[dict]) -> str:
    parts = []

    document_context = build_chunk_context(chunks)
    if document_context:
        parts.append("### DOCUMENT EXCERPTS\n\n" + document_context)

    graph_context = build_graph_context(graph_facts)
    if graph_context:
        parts.append("### KNOWLEDGE GRAPH FACTS\n\n" + graph_context)

    return "\n\n".join(parts) if parts else "(no evidence retrieved)"


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


def classify_scope(client: Groq, question: str, context: str) -> bool:
    try:
        response = _create_completion(
            client,
            model=MODEL_NAME,
            messages=[{"role": "user", "content": SCOPE_PROMPT.format(question=question, context=context)}],
            temperature=0,
            reasoning_effort="low",
            max_completion_tokens=150,
        )
        text = (response.choices[0].message.content or "").strip().upper()
        verdict_line = next((line for line in text.splitlines() if "VERDICT" in line), text)
        return "YES" in verdict_line
    except Exception as exc:
        print(f"[DEBUG] Scope classification failed: {exc!r}")
        return True


def is_in_scope(client: Groq, question: str, context: str, top_reranker_score: float) -> bool:
    retrieval_is_weak = top_reranker_score < SCOPE_CONFIDENCE_FLOOR

    if not retrieval_is_weak:
        return True

    llm_says_in_scope = classify_scope(client, question, context)
    return not (not llm_says_in_scope and retrieval_is_weak)


# ==========================================================================
# Step 4 — Draft
# ==========================================================================

DRAFT_PROMPT = """
You are a hepatology research assistant. Answer the question STRICTLY
using the evidence below (document excerpts tagged [Source: <section>,
p.<page>] and knowledge graph facts tagged [Graph source: <section>,
p.<page>]). Never use outside knowledge, and cite every claim with the
exact tag it came from.

If the evidence doesn't contain the answer, reply with exactly:
I don't know based on the available sources.

QUESTION:
{question}

EVIDENCE:
{context}
"""


def draft_answer(client: Groq, query: str, context: str) -> str:
    prompt = DRAFT_PROMPT.format(question=query, context=context)

    return _create_completion_with_retry(
        client,
        step_name="draft_answer",
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=2000,
    )


# ==========================================================================
# Step 5 — Verification
# ==========================================================================

VERIFY_PROMPT = """
You are a strict fact-checker. Check the DRAFT ANSWER against the
EVIDENCE — every claim and citation must be literally backed by it.
Fix or remove anything that isn't. If nothing supported remains, output
exactly: I don't know based on the available sources.

Output ONLY the final answer text, no commentary.

QUESTION:
{question}

EVIDENCE:
{context}

DRAFT ANSWER:
{draft}
"""


def verify_answer(client: Groq, query: str, context: str, draft: str) -> str:
    prompt = VERIFY_PROMPT.format(question=query, context=context, draft=draft)

    return _create_completion_with_retry(
        client,
        step_name="verify_answer",
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=2500,
    )


def _extract_citations(text: str) -> list[tuple[str, str, str]]:
    return [(kind.strip(), section.strip(), page.strip()) for kind, section, page in CITATION_RE.findall(text)]


def _valid_citation_keys(chunks: list[dict], graph_facts: list[dict]) -> set[tuple[str, str, str]]:
    valid = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}
        section = str(metadata.get("section", "Unknown")).strip()
        page = str(metadata.get("page_start", "?")).strip()
        valid.add(("Source", section, page))

    for fact in graph_facts:
        section = str(fact.get("section") or "Unknown").strip()
        page = str(fact.get("page") or "?").strip()
        valid.add(("Graph source", section, page))

    return valid


def has_only_verifiable_citations(answer: str, chunks: list[dict], graph_facts: list[dict]) -> bool:
    citations = _extract_citations(answer)

    if not citations:
        return True

    valid_keys = _valid_citation_keys(chunks, graph_facts)
    return all(citation in valid_keys for citation in citations)


def finalize_answer(answer: str, top_reranker_score: float) -> str:
    confidence_note = ""

    if top_reranker_score < LOW_CONFIDENCE_THRESHOLD:
        confidence_note = (
            "\n\n⚠️ *Low confidence retrieval — the closest match found "
            "wasn't a strong fit for this question.*"
        )

    return answer + confidence_note + DISCLAIMER


# ==========================================================================
# Main answer chain
# ==========================================================================

def answer_question(client: Groq, driver, query: str, top_k: int = RETRIEVAL_TOP_K) -> str:
    if is_blocked(query):
        return (
            "I can't give personal medical advice "
            "(dosage, prescriptions, or diagnosis). "
            "I can share what the research paper says "
            "about liver disease topics in general."
        )

    t_start = time.time()

    t0 = time.time()
    search_query = rewrite_query(client, query)
    print(f"[TIMING] rewrite_query: {time.time() - t0:.2f}s")
    print(f"[DEBUG] Search query: {search_query}")

    t0 = time.time()
    chunks = retrieve_chunks(search_query, top_k=top_k)
    print(f"[TIMING] retrieve_chunks (semantic+bm25+rrf+rerank): {time.time() - t0:.2f}s")

    t0 = time.time()
    graph_facts = retrieve_graph_facts(driver, query, limit=GRAPH_FACTS_LIMIT)
    print(f"[TIMING] retrieve_graph_facts: {time.time() - t0:.2f}s")

    print("\n=== Graph facts ===")
    if not graph_facts:
        print("(none retrieved)")
    else:
        for fact in graph_facts:
            section = fact.get("section") or "Unknown"
            page = fact.get("page") or "?"
            print(f"{fact.get('source')} -[{fact.get('relation')}]-> {fact.get('target')} "
                  f"(section={section}, p.{page})")

    if not chunks and not graph_facts:
        return NOT_FOUND_TEXT + DISCLAIMER

    context = build_full_context(chunks, graph_facts)
    top_reranker_score = float(chunks[0].get("reranker_score", 1.0)) if chunks else 0.0

    t0 = time.time()
    in_scope = is_in_scope(client, query, context, top_reranker_score)
    print(f"[TIMING] is_in_scope: {time.time() - t0:.2f}s")

    if not in_scope:
        return OUT_OF_SCOPE_TEXT + DISCLAIMER

    try:
        t0 = time.time()
        draft = draft_answer(client, query, context)
        print(f"[TIMING] draft_answer: {time.time() - t0:.2f}s")

        t0 = time.time()
        verified = verify_answer(client, query, context, draft)
        print(f"[TIMING] verify_answer: {time.time() - t0:.2f}s")

    except GenerationError as exc:
        print(f"[DEBUG] GenerationError: {exc}")
        return UNVERIFIED_TEXT + DISCLAIMER

    if not has_only_verifiable_citations(verified, chunks, graph_facts):
        print("[DEBUG] Citation validation failed.")
        print("--- DRAFT ---")
        print(draft)
        print("--- VERIFIED ---")
        print(verified)
        print("--- VALID CITATIONS ---")
        print(_valid_citation_keys(chunks, graph_facts))
        print("--- FOUND CITATIONS ---")
        print(_extract_citations(verified))
        return UNVERIFIED_TEXT + DISCLAIMER

    print(f"[TIMING] TOTAL (answer_question): {time.time() - t_start:.2f}s")
    return finalize_answer(verified, top_reranker_score)


def main():
    t_process_start = time.time()

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

    query = " ".join(sys.argv[1:]) or "What causes liver cirrhosis?"
    print(f"Query: {query}\n")

    client = Groq(api_key=api_key)
    driver = get_neo4j_driver()

    if driver is None:
        print("(Neo4j unavailable — answering from vector/BM25 only.)\n")

    try:
        answer = answer_question(client, driver, query)
    finally:
        if driver is not None:
            driver.close()

    print(f"[TIMING] TOTAL (whole process, incl. imports): {time.time() - t_process_start:.2f}s")
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()
