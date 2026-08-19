"""Step 3a: chunks.json -> embeddings.npy + embeddings_meta.json

Run: python rag/embed.py
"""
import json
import os
import pathlib
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import (
    CHUNKS_PATH, EMBEDDINGS_PATH as EMBEDDINGS_OUT_PATH,
    EMBEDDINGS_META_PATH as META_OUT_PATH, EMBED_MODEL_NAME, EMBED_BATCH_SIZE as BATCH_SIZE,
)


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Could not find {CHUNKS_PATH}. Run rag/chunker.py first.")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("chunks.json is empty.")

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{c['chunk_id']}" for c in chunks]

    print("Computing embeddings (this can take a while on CPU)...")
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)

    if len(embeddings) != len(chunks):
        raise RuntimeError("Number of embeddings does not match number of chunks.")

    np.save(EMBEDDINGS_OUT_PATH, embeddings)

    meta = [{
        "chunk_id": c["chunk_id"],
        "id": ids[i],
        "text": c["text"],
        "source": c.get("source", ""),
        "doi": c.get("doi", ""),
        "section": c.get("section", "Unknown"),
        "page_start": c.get("page_start"),
        "page_end": c.get("page_end"),
        "n_tokens": c.get("n_tokens"),
    } for i, c in enumerate(chunks)]

    META_OUT_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved embeddings array {embeddings.shape} -> {EMBEDDINGS_OUT_PATH}")
    print(f"Saved matching metadata -> {META_OUT_PATH}")


if __name__ == "__main__":
    main()
