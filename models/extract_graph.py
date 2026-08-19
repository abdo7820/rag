"""
models/extract_graph.py

Step 4a: chunks.json -> data/graph_triples.json

Uses Groq (LLM) to pull out medical entities and relationships from each
chunk. Every triple keeps the chunk_id it came from, so the graph stays
traceable back to a page/section/DOI, same as the vector + BM25 retrieval.

Resumable: progress is saved after every chunk, and chunks already present
in data/graph_triples.json are skipped on the next run.

Run:
    python models/extract_graph.py
"""
import json
import os
import pathlib
import re
import sys
import time

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import CHUNKS_PATH, GRAPH_TRIPLES_PATH as OUT_PATH, GRAPH_EXTRACTION_MODEL_NAME as MODEL_NAME

MAX_RETRIES = 3
RATE_LIMIT_WAIT_PATTERN = re.compile(r"try again in ([\d.]+)m([\d.]+)s")

ENTITY_TYPES = [
    "DISEASE", "SYMPTOM", "RISK_FACTOR", "DRUG", "TREATMENT",
    "GENE", "BIOMARKER", "ORGANIZATION", "POPULATION",
]

SYSTEM_PROMPT = f"""You are a biomedical information extraction system.
Given a chunk of text from a hepatology (liver disease) research paper,
extract entities and the relationships between them.

Allowed entity types: {", ".join(ENTITY_TYPES)}

Respond with ONLY valid JSON (no markdown fences, no commentary) in this
exact shape:

{{
  "entities": [
    {{"name": "Non-alcoholic fatty liver disease", "type": "DISEASE"}}
  ],
  "relationships": [
    {{"source": "Obesity", "relation": "INCREASES_RISK_OF", "target": "Non-alcoholic fatty liver disease"}}
  ]
}}

Rules:
- Only extract entities/relationships that are explicitly stated in the text.
- Use short, canonical entity names (no full sentences).
- If nothing relevant is found, return {{"entities": [], "relationships": []}}.
- Do not invent facts not present in the text.
"""


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Could not find {CHUNKS_PATH}. Run rag/chunker.py first.")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("chunks.json is empty.")
    return chunks


def load_existing_records() -> dict:
    if not OUT_PATH.exists():
        return {}
    try:
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        return {r["chunk_id"]: r for r in existing}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_records(records: list[dict]):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json_response(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def rate_limit_wait_seconds(error_message: str) -> float | None:
    match = RATE_LIMIT_WAIT_PATTERN.search(error_message)
    if not match:
        return None
    minutes, seconds = match.groups()
    return float(minutes) * 60 + float(seconds)


def extract_from_chunk(client: Groq, chunk: dict) -> dict | None:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chunk["text"]},
                ],
                temperature=0,
            )
            parsed = parse_json_response(response.choices[0].message.content)
            parsed.setdefault("entities", [])
            parsed.setdefault("relationships", [])
            return parsed
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if "rate_limit_exceeded" in message and "tokens per day" in message:
                wait = rate_limit_wait_seconds(message)
                print(f"\n  [rate limit] Daily token limit hit on chunk {chunk['chunk_id']}.")
                if wait:
                    print(f"  Groq says try again in ~{wait / 60:.1f} min. Stopping for now — re-run this "
                          f"script later and it will resume from here.")
                else:
                    print("  Stopping for now — re-run this script later and it will resume from here.")
                return None
            time.sleep(1.5 * attempt)

    print(f"  [warn] chunk {chunk['chunk_id']}: giving up after {MAX_RETRIES} tries ({last_error})")
    return {"entities": [], "relationships": []}


def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

    chunks = load_chunks()
    existing = load_existing_records()
    remaining = [c for c in chunks if c["chunk_id"] not in existing]

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    if existing:
        print(f"Resuming: {len(existing)} chunks already done, {len(remaining)} left.")
    print(f"Extracting entities/relationships with {MODEL_NAME} ...")

    if not remaining:
        print("\nNothing left to do — all chunks already processed.")
        return

    client = Groq(api_key=api_key)
    records = existing.copy()

    for i, chunk in enumerate(remaining, start=1):
        result = extract_from_chunk(client, chunk)
        if result is None:
            break

        records[chunk["chunk_id"]] = {
            "chunk_id": chunk["chunk_id"],
            "section": chunk.get("section"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "doi": chunk.get("doi"),
            "entities": result["entities"],
            "relationships": result["relationships"],
        }

        save_records(sorted(records.values(), key=lambda r: r["chunk_id"]))

        if i % 10 == 0 or i == len(remaining):
            print(f"  {len(existing) + i}/{len(chunks)} chunks processed")

    final_records = sorted(records.values(), key=lambda r: r["chunk_id"])
    total_entities = sum(len(r["entities"]) for r in final_records)
    total_relationships = sum(len(r["relationships"]) for r in final_records)

    print(f"\nSaved graph triples -> {OUT_PATH}")
    print(f"Chunks done: {len(final_records)}/{len(chunks)}")
    print(f"Total entities extracted (with duplicates): {total_entities}")
    print(f"Total relationships extracted: {total_relationships}")

    if len(final_records) < len(chunks):
        print("\nSome chunks are still pending (rate limit). Re-run this script later to finish the rest.")


if __name__ == "__main__":
    main()
