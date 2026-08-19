"""Step 1: PDF -> Markdown (pip install pymupdf4llm)

Run: python rag/pdf_to_markdown.py
"""
import pathlib
import sys

import pymupdf4llm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import PDF_PATH, MD_PATH


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Could not find {PDF_PATH}. Place the source PDF there first.")

    md_text = pymupdf4llm.to_markdown(str(PDF_PATH))
    MD_PATH.write_text(md_text, encoding="utf-8")
    print(f"Extracted {len(md_text)} characters -> {MD_PATH}")


if __name__ == "__main__":
    main()
