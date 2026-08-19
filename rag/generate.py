"""
rag/generate.py

A 6-step prompt chain that turns a question into a grounded, cited answer,
pulling evidence from BOTH the vector/BM25 index and the Neo4j graph.

The chain:
  1. Guardrail          — reject personal medical-advice questions (deterministic)
  2. Query rewrite       — LLM call: expand abbreviations/synonyms into a
                           sharper search query, to improve retrieval recall
  3. Retrieve            — Chroma+BM25 -> RRF -> rerank (using the rewritten
                           query), AND Neo4j graph facts (using the original
                           question, so entity names still match verbatim)
  4. Scope check          — LLM call (with reasoning) + a deterministic
                           retrieval-confidence signal, combined so a single
                           bad LLM call can't wrongly block or allow an answer
  5. Draft answer         — LLM call: answer strictly from the retrieved
                           context, with inline citations
  6. Verify & finalize    — LLM call: re-check the draft against the context,
                           strip any unsupported claim/citation, then attach
                           the confidence indicator + disclaimer (deterministic)

Run:
    python rag/generate.py "What causes liver cirrhosis?"
"""
import os
import pathlib
import re
import sys
import time

try:
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
except ImportError:
    # torch isn't installed on light deployments (requirements-railway.txt +
    # USE_HF_INFERENCE_API=true) — embedding/reranking run via HTTP APIs
    # instead, so this is fine. Without this guard, importing generate.py
    # (and therefore app.py, which imports it) crashes on any container
    # that only installed requirements-railway.txt.
    print("[DEBUG] torch not installed — assuming USE_HF_INFERENCE_API=true deployment.")

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "rag"))
sys.path.insert(0, str(BASE_DIR))

from test_search import semantic_search, bm25_search, reciprocal_rank_fusion, rerank_results  # noqa: E402
from config import (  # noqa: E402
    GENERATION_MODEL_NAME as MODEL_NAME,
    LOW_CONFIDENCE_THRESHOLD,
    SCOPE_CONFIDENCE_FLOOR,
    GRAPH_FACTS_LIMIT,
    RETRIEVAL_TOP_K,
    GENERATION_MAX_RETRIES,
    GENERATION_RETRY_BACKOFF_S,
)

# NOTE: llama-3.3-70b-versatile / llama-3.1-8b-instant are deprecated on Groq
# (announced 2026-06-17) — do not switch back to them, calls will 404.
# Official replacements are openai/gpt-oss-120b or qwen/qwen3.6-27b.

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

CITATION_RE = re.compile(
    r"\[(Graph source|Source):\s*([^,\]]+?)\s*,\s*p\.\s*([\w\-]+)\s*\]"
)


class GenerationError(RuntimeError):
    """Raised when a required LLM generation step (draft/verify) could not
    produce usable output after retries. Callers must NOT fall back to an
    unverified answer when this is raised."""


# ============================================================
# Step 1 — Guardrail (deterministic, no LLM call)
# ============================================================

BLOCKED_PATTERNS = re.compile(
    r"\b(dosage|dose|prescri\w*|should i take|how much .*(should|do i) take|"
    r"is it safe for me|diagnose me|do i have)\b",
    re.IGNORECASE,
)

DISCLAIMER = (
    "\n\n---\n"
    "*This is educational information drawn from a research paper, not "
    "medical advice. Please consult a qualified healthcare professional "
    "for any personal medical decisions.*"
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
    last_error: Exception | None = None

    for attempt in range(1, GENERATION_MAX_RETRIES + 1):
        try:
            response = _create_completion(client, **kwargs)
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
            last_error = RuntimeError(f"{step_name}: model returned an empty response")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[DEBUG] {step_name} attempt {attempt}/{GENERATION_MAX_RETRIES} failed: {exc!r}")

        if attempt < GENERATION_MAX_RETRIES:
            time.sleep(GENERATION_RETRY_BACKOFF_S * attempt)

    raise GenerationError(
        f"{step_name} failed after {GENERATION_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


# ============================================================
# Step 2 — Query rewrite (LLM call #1)
# ============================================================

REWRITE_PROMPT = """Rewrite the question below into a focused search query \
for retrieving passages from a hepatology (liver disease) research paper.

- Expand any medical abbreviations to their full form alongside the \
  abbreviation (e.g. "HBV" -> "HBV hepatitis B virus").
- Add closely related medical synonyms if they would help retrieval.
- Keep it short: a search query, not a sentence or an answer.
- Do not answer the question. Do not add commentary.

Question: {question}

Search query:"""


def rewrite_query(client: Groq, question: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
            temperature=0,
            reasoning_effort="low",
            max_completion_tokens=200,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten if rewritten else question
    except Exception:
        return question


# ============================================================
# Step 3 — Retrieve (vector + BM25 + graph)
# ============================================================

def retrieve_chunks(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    semantic_results = semantic_search(query)
    bm25_results = bm25_search(query)
    fused = reciprocal_rank_fusion(semantic_results, bm25_results)
    reranked = rerank_results(query, fused)
    return reranked[:top_k]


def build_chunk_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        meta = c.get("metadata", {})
        section = meta.get("section", "Unknown")
        page = meta.get("page_start", "?")
        blocks.append(f"[Source: {section}, p.{page}]\n{c['text']}")
    return "\n\n".join(blocks)


def get_neo4j_driver():
    """Returns a connected Neo4j driver, or None if credentials are missing
    OR the connection can't actually be established (e.g. a paused Aura
    instance). Verifying connectivity here matters — a driver object is
    created successfully even when unreachable, and every later query
    against it would otherwise fail silently. Fail loud here instead, once.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, password]):
        return None

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
    except Exception as exc:
        print(f"[warn] Neo4j is configured but unreachable ({exc}). "
              f"Continuing WITHOUT graph facts — answers will only use the "
              f"vector/BM25 index. If this is an Aura free-tier instance, "
              f"it may just be paused: resume it in the Neo4j Aura console "
              f"and re-run.")
        driver.close()
        return None

    return driver


def _dedupe_graph_facts(facts: list[dict]) -> list[dict]:
    """Collapses facts identical from the reader's point of view — same
    (source, relation, target, section, page) — down to one entry. This is
    a display/context-building dedupe only; chunker.py's 80-token overlap
    means the same real fact can legitimately be extracted from more than
    one chunk, and those stay in the database as distinct citations.
    """
    seen = set()
    deduped = []
    for f in facts:
        key = (f.get("source"), f.get("relation"), f.get("target"), f.get("section"), f.get("page"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    return deduped


# Cached once per process — the entity vocabulary doesn't change at runtime,
# so there's no need to re-fetch it on every question. Restart the server
# after re-running models/load_graph.py to pick up new entities.
_all_entity_names: list[str] | None = None


def _get_all_entity_names(driver, database: str) -> list[str]:
    global _all_entity_names
    if _all_entity_names is None:
        with driver.session(database=database) as session:
            result = session.run(
                "MATCH (e:Entity) WHERE size(e.name) > 2 RETURN DISTINCT e.name AS name"
            )
            _all_entity_names = [r["name"] for r in result]
    return _all_entity_names


def _find_mentioned_entities(question: str, all_names: list[str]) -> list[str]:
    """Matches entity names against the question using word boundaries, so
    a short entity like "AST" doesn't false-positive match inside an
    unrelated word like "contrast" or "fasting" — which plain substring
    matching (the old Cypher `CONTAINS` approach) would have allowed.
    """
    question_lower = question.lower()
    matched = []
    for name in all_names:
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        if re.search(pattern, question_lower):
            matched.append(name)
    return matched


def retrieve_graph_facts(driver, original_question: str, limit: int = GRAPH_FACTS_LIMIT) -> list[dict]:
    """Find entities mentioned in the ORIGINAL question (not the rewritten
    query — entity names in the graph must match the user's own wording,
    e.g. "HBV" not "HBV hepatitis B virus"), then pull their relationships.
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
                RETURN a.name AS source, r.relation AS relation, b.name AS target,
                       r.section AS section, r.page_start AS page
                LIMIT $limit
                """,
                names=names, limit=limit,
            )
            facts = [dict(record) for record in result]
            return _dedupe_graph_facts(facts)
    except Exception as exc:
        print(f"[warn] Neo4j query failed mid-session ({exc}). Answering from vector/BM25 only for this question.")
        return []


def build_graph_context(facts: list[dict]) -> str:
    if not facts:
        return ""
    lines = []
    for f in facts:
        section = f.get("section") or "Unknown"
        page = f.get("page") or "?"
        lines.append(f"[Graph source: {section}, p.{page}] {f['source']} -[{f['relation']}]-> {f['target']}")
    return "\n".join(lines)


def build_full_context(chunks: list[dict], graph_facts: list[dict]) -> str:
    parts = []
    chunk_context = build_chunk_context(chunks)
    if chunk_context:
        parts.append("### DOCUMENT EXCERPTS\n\n" + chunk_context)
    graph_context = build_graph_context(graph_facts)
    if graph_context:
        parts.append("### KNOWLEDGE GRAPH FACTS\n\n" + graph_context)
    return "\n\n".join(parts) if parts else "(no evidence retrieved)"


# ============================================================
# Step 4 — Scope check (LLM call #2 + deterministic retrieval signal)
# ============================================================

SCOPE_PROMPT = """You will be shown a QUESTION and EVIDENCE excerpts \
retrieved for it from a hepatology (liver disease) research paper.

Think briefly about whether this question is a hepatology/liver-disease- \
related medical or scientific question, AND whether the evidence is at \
least topically related to it — not whether the evidence fully answers it, \
just whether it's plausibly on the same topic.

Respond in exactly this format, two lines:
Reasoning: <one short sentence>
Verdict: YES or NO

QUESTION: {question}

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
    except Exception:
        return True


def is_in_scope(client: Groq, question: str, context: str, top_reranker_score: float) -> bool:
    retrieval_is_weak = top_reranker_score < SCOPE_CONFIDENCE_FLOOR
    print(f"[DEBUG] top_reranker_score={top_reranker_score:.4f} "
          f"SCOPE_CONFIDENCE_FLOOR={SCOPE_CONFIDENCE_FLOOR} weak={retrieval_is_weak}")
    if not retrieval_is_weak:
        return True

    llm_says_in_scope = classify_scope(client, question, context)
    result = not (not llm_says_in_scope and retrieval_is_weak)
    if not result:
        print(f"[DEBUG] is_in_scope=False -> returning OUT_OF_SCOPE_TEXT "
              f"(retrieval weak AND scope classifier said NO)")
    return result


# ============================================================
# Step 5 — Draft answer (LLM call #3)
# ============================================================

DRAFT_PROMPT = """You are a hepatology research assistant answering questions \
strictly from the material provided below. You have two kinds of evidence:

1. DOCUMENT EXCERPTS — passages retrieved from a research paper, each tagged \
   with [Source: <section>, p.<page>].
2. KNOWLEDGE GRAPH FACTS — entity relationships extracted from the same \
   paper, each tagged with [Graph source: <section>, p.<page>].

Follow these rules exactly, in order:

1. Read all the evidence below before answering.
2. Only state something if it is directly supported by the evidence. Never \
   use outside/prior knowledge, even if you are confident it's correct.
3. After every factual claim, add its citation in the EXACT format \
   [Source: <section>, p.<page>] or [Graph source: <section>, p.<page>] — \
   never drop the page number, never shorten "Graph source" to "Source", \
   and never omit either part.
4. If the evidence answers the question only partially, answer only the \
   part that is supported, and explicitly say which part is not covered.
5. If the evidence does not contain the answer at all, respond with EXACTLY \
   this sentence and nothing else: "{not_found}"
6. Never fabricate a citation, a page number, a statistic, or a relationship \
   that is not literally present in the evidence below.
7. Prefer combining both DOCUMENT EXCERPTS and KNOWLEDGE GRAPH FACTS when \
   both are relevant — they reinforce each other and come from the same paper.
8. Be concise. Do not repeat the question back. Do not add disclaimers — \
   those are added separately.

EVIDENCE:
{context}
""".replace("{not_found}", NOT_FOUND_TEXT)


def draft_answer(client: Groq, query: str, context: str) -> str:
    system_prompt = DRAFT_PROMPT.format(context=context)
    return _create_completion_with_retry(
        client,
        step_name="draft_answer",
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=2000,
    )


# ============================================================
# Step 6a — Verify (LLM call #4): catch unsupported claims/citations
# ============================================================

VERIFY_PROMPT = """You are a strict fact-checker. Below is a QUESTION, the \
EVIDENCE that was available, and a DRAFT ANSWER written from that evidence.

Check the draft against the evidence, claim by claim:
- Every factual statement in the draft must be backed by something literally \
  present in the evidence.
- Every citation must correctly reference a source that actually appears in \
  the evidence, and must attribute the right fact to the right source.
- Every citation must be in the EXACT format [Source: <section>, p.<page>] \
  or [Graph source: <section>, p.<page>] — both the section AND the page \
  number are required every time. If a citation in the draft is missing the \
  page number or the "Graph" prefix where it applies, fix its formatting to \
  match the evidence block it came from.

If the draft is fully supported, output it UNCHANGED, word for word.

If any part of the draft is NOT supported, rewrite the draft to remove or \
correct only that part, keeping everything else and all valid citations \
intact. If removing the unsupported part leaves nothing left, output \
exactly: "{not_found}"

Output ONLY the final answer text — no commentary, no explanation of what \
you changed, no preamble like "Here is the corrected answer".

QUESTION:
{question}

EVIDENCE:
{context}

DRAFT ANSWER:
{draft}
""".replace("{not_found}", NOT_FOUND_TEXT)


def verify_answer(client: Groq, query: str, context: str, draft: str) -> str:
    return _create_completion_with_retry(
        client,
        step_name="verify_answer",
        model=MODEL_NAME,
        messages=[{"role": "user", "content": VERIFY_PROMPT.format(question=query, context=context, draft=draft)}],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=2500,
    )


def _normalize_citation_key(kind: str, section: str, page: str) -> tuple[str, str, str]:
    """Normalizes a citation key so that harmless formatting differences —
    extra whitespace, case, a trailing '.0' on a page number that came from
    a float/Decimal in the DB, etc. — don't cause a real citation to be
    treated as unverifiable. This matters because production and local can
    end up with metadata that is semantically identical but not byte-identical
    (e.g. Chroma returning page_start as 12.0 instead of "12").
    """
    kind_norm = kind.strip().lower()
    section_norm = re.sub(r"\s+", " ", section.strip()).lower()
    page_norm = page.strip().lower()
    if page_norm.endswith(".0"):
        page_norm = page_norm[:-2]
    return (kind_norm, section_norm, page_norm)


def _extract_citations(text: str) -> list[tuple[str, str, str]]:
    return [(kind.strip(), section.strip(), page.strip()) for kind, section, page in CITATION_RE.findall(text)]


def _valid_citation_keys(chunks: list[dict], graph_facts: list[dict]) -> set[tuple[str, str, str]]:
    valid = set()
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        section = str(meta.get("section", "Unknown")).strip()
        page = str(meta.get("page_start", "?")).strip()
        valid.add(_normalize_citation_key("Source", section, page))
    for f in graph_facts:
        section = str(f.get("section") or "Unknown").strip()
        page = str(f.get("page") or "?").strip()
        valid.add(_normalize_citation_key("Graph source", section, page))
    return valid


def has_only_verifiable_citations(answer: str, chunks: list[dict], graph_facts: list[dict]) -> bool:
    citations = _extract_citations(answer)
    if not citations:
        return True
    valid_keys = _valid_citation_keys(chunks, graph_facts)
    return all(
        _normalize_citation_key(kind, section, page) in valid_keys
        for kind, section, page in citations
    )


# ============================================================
# Step 6b — Finalize (deterministic, no LLM call)
# ============================================================

def finalize_answer(answer: str, top_reranker_score: float) -> str:
    confidence_note = ""
    if top_reranker_score < LOW_CONFIDENCE_THRESHOLD:
        confidence_note = "\n\n⚠️ *Low confidence retrieval — the closest match found wasn't a strong fit for this question.*"
    return answer + confidence_note + DISCLAIMER


# ============================================================
# Chain orchestration
# ============================================================

def answer_question(client: Groq, driver, query: str, top_k: int = RETRIEVAL_TOP_K) -> str:
    if is_blocked(query):
        return (
            "I can't give personal medical advice (dosage, prescriptions, or "
            "diagnosis) — please talk to a doctor or pharmacist about that. "
            "I can share what the research paper says about liver disease "
            "topics in general, if that helps."
        )

    t_start = time.time()

    t0 = time.time()
    search_query = rewrite_query(client, query)
    print(f"[TIMING] rewrite_query: {time.time() - t0:.2f}s")

    t0 = time.time()
    chunks = retrieve_chunks(search_query, top_k=top_k)
    print(f"[TIMING] retrieve_chunks (semantic+bm25+rrf+rerank): {time.time() - t0:.2f}s")

    t0 = time.time()
    graph_facts = retrieve_graph_facts(driver, query)
    print(f"[TIMING] retrieve_graph_facts: {time.time() - t0:.2f}s")

    print("\n=== Graph facts ===")
    if not graph_facts:
        print("(none retrieved — either no matching entities, or Neo4j is unavailable for this run)")
    else:
        for f in graph_facts:
            section = f.get("section") or "Unknown"
            page = f.get("page") or "?"
            print(f"{f['source']} -[{f['relation']}]-> {f['target']}  (section={section}, p.{page})")

    if not chunks and not graph_facts:
        return NOT_FOUND_TEXT + DISCLAIMER

    context = build_full_context(chunks, graph_facts)
    top_reranker_score = chunks[0].get("reranker_score", 1.0) if chunks else 0.0

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
        print(f"[DEBUG] Citation check failed.\n--- draft ---\n{draft}\n--- verified ---\n{verified}\n"
              f"--- valid keys ---\n{_valid_citation_keys(chunks, graph_facts)}\n"
              f"--- citations found ---\n{_extract_citations(verified)}")
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

    print(f"[DEBUG] Using GENERATION_MODEL_NAME = {MODEL_NAME!r}")
    if MODEL_NAME in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        print("[WARN] This model is deprecated on Groq (as of 2026-06-17) and "
              "calls will 404 — update GENERATION_MODEL_NAME in config.py to "
              "openai/gpt-oss-120b or qwen/qwen3.6-27b.")

    client = Groq(api_key=api_key)
    driver = get_neo4j_driver()
    if driver is None:
        print("(Neo4j credentials not found in .env — answering from vector/BM25 only.)\n")

    try:
        answer = answer_question(client, driver, query)
    finally:
        if driver is not None:
            driver.close()

    print(f"[TIMING] TOTAL (whole process, incl. model loading + imports): {time.time() - t_process_start:.2f}s")

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()
