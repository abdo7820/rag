"""Step 3c: chunks.json -> BM25 index (data/bm25_index.pkl)

Run: python rag/bm25_index.py
"""
import json
import pathlib
import pickle
import re
import sys

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import CHUNKS_PATH, BM25_INDEX_PATH as INDEX_OUT_PATH

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def simple_tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Could not find {CHUNKS_PATH}. Run rag/chunker.py first.")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("chunks.json is empty.")

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    print("Building BM25 index...")

    tokenized_corpus = [simple_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    metadata = [{
        "chunk_id": c["chunk_id"],
        "source": c.get("source", ""),
        "doi": c.get("doi", ""),
        "section": c.get("section", ""),
        "page_start": c.get("page_start"),
        "page_end": c.get("page_end"),
        "n_tokens": c.get("n_tokens"),
    } for c in chunks]

    payload = {
        "bm25": bm25,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "texts": [c["text"] for c in chunks],
        "metadata": metadata,
    }

    INDEX_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_OUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"BM25 index built and saved to {INDEX_OUT_PATH}")
    print(f"Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    main()
