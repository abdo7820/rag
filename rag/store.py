"""Step 3b: embeddings.npy -> persistent Chroma store (vectorDB/)

Run: python rag/store.py
"""
import json
import pathlib
import sys

import chromadb
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import (
    EMBEDDINGS_PATH, EMBEDDINGS_META_PATH as META_PATH,
    VECTOR_DB_DIR, COLLECTION_NAME, CHROMA_INSERT_BATCH as BATCH,
)


def to_chroma_metadata(meta):
    """Chroma only accepts primitive scalars; None -> -1 for numeric fields."""
    return [{
        "chunk_id": int(m["chunk_id"]),
        "source": str(m.get("source", "")),
        "doi": str(m.get("doi", "")),
        "section": str(m.get("section", "")),
        "page_start": int(m["page_start"]) if m.get("page_start") is not None else -1,
        "page_end": int(m["page_end"]) if m.get("page_end") is not None else -1,
        "n_tokens": int(m["n_tokens"]) if m.get("n_tokens") is not None else -1,
    } for m in meta]


def main():
    if not EMBEDDINGS_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Missing embeddings/metadata. Run rag/embed.py first.")

    embeddings = np.load(EMBEDDINGS_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    if len(embeddings) != len(meta):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(meta)} metadata records.")

    print(f"Loaded {embeddings.shape[0]} embeddings from {EMBEDDINGS_PATH}")

    ids = [str(m["id"]) for m in meta]
    texts = [str(m["text"]) for m in meta]
    metadatas = to_chroma_metadata(meta)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Deleting existing collection '{COLLECTION_NAME}' ...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    embeddings_list = embeddings.tolist()

    try:
        max_batch = client.get_max_batch_size()
        batch_size = min(BATCH, max_batch)
    except Exception:
        batch_size = BATCH

    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings_list[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"Inserted records {start} -> {min(end, len(texts))}")

    print(f"\nStored {collection.count()} vectors in collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
