"""Step 2: Markdown -> chunks.json, keeps page + section metadata.

Run: python rag/chunker.py
"""
import json
import pathlib
import re
import sys

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import MD_PATH, CHUNKS_PATH as OUT_PATH, SOURCE_NAME, DOI, CHUNK_SIZE, CHUNK_OVERLAP

# Footer marks the end of each printed page, e.g.:
# "Signal Transduction and Targeted Therapy (2025) 10:33 Liver diseases... Gan et al. 3"
# NOTE: this pattern is specific to THIS paper's running footer. If you point
# the pipeline at a different PDF, update (or parametrize) this regex first —
# otherwise every chunk's page_start/page_end will silently fall back to 1.
PAGE_FOOTER = re.compile(
    r"Signal\s+Transduction\s+and\s+Targeted\s+Therapy\s+\(2025\)\s+10:33\s+"
    r"Liver\s+diseases:\s+epidemiology,\s+causes,\s+trends\s+and\s+predictions\s+Gan\s+et\s+al\.\s+(\d+)",
    re.IGNORECASE,
)
SECTION_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")


def build_boundaries(text):
    pages = [{"pos": m.end(), "page": int(m.group(1))} for m in PAGE_FOOTER.finditer(text)]

    sections = []
    for m in SECTION_HEADING.finditer(text):
        heading = m.group(2).strip()
        if heading.startswith("Liver diseases: epidemiology") or heading == "REVIEW ARTICLE **OPEN**":
            continue
        sections.append({"pos": m.start(), "section": heading})

    return pages, sections


def lookup(boundaries, pos, key, default):
    current = default
    for b in boundaries:
        if b["pos"] <= pos:
            current = b[key]
        else:
            break
    return current


def main():
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Could not find markdown file: {MD_PATH}. Run rag/pdf_to_markdown.py first.")

    text = MD_PATH.read_text(encoding="utf-8")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )

    pages, sections = build_boundaries(text)
    documents = splitter.create_documents([text])

    records = []
    for i, doc in enumerate(documents):
        start = doc.metadata.get("start_index", 0)
        end = start + len(doc.page_content)

        records.append({
            "chunk_id": i,
            "text": doc.page_content,
            "n_tokens": len(tokenizer.encode(doc.page_content)),
            "source": SOURCE_NAME,
            "doi": DOI,
            "section": lookup(sections, start, "section", "Introduction"),
            "page_start": lookup(pages, start, "page", 1),
            "page_end": lookup(pages, end, "page", 1),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_tokens = sum(r["n_tokens"] for r in records) / len(records) if records else 0
    print(f"Produced {len(records)} chunks -> {OUT_PATH}")
    print(f"Avg tokens/chunk: {avg_tokens:.1f}")


if __name__ == "__main__":
    main()
