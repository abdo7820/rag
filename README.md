# 🩺 Liver Diseases RAG

### Production-Ready Hybrid RAG System for Hepatology Research

> **Semantic Retrieval + BM25 + RRF Fusion + Cross-Encoder Reranking + Neo4j Knowledge Graph + Grounded Generation + Self-Verification**

A production-oriented Retrieval-Augmented Generation (RAG) system for question answering over the research paper:

**“Liver diseases: epidemiology, causes, trends and predictions”**  
*Signal Transduction and Targeted Therapy* — DOI: `10.1038/s41392-024-02072-z`

The system combines **dense semantic retrieval**, **lexical retrieval**, **knowledge-graph context**, **reranking**, and a **6-step grounded generation chain** to produce answers that stay anchored to the source document.

---

## 🏗️ System Architecture

<img width="1536" height="1024" alt="ChatGPT Image Aug 19, 2026, 07_52_28 PM" src="https://github.com/user-attachments/assets/e79ad3ec-6409-4352-84c6-3fa26a715c57" />



### End-to-End Flow

```text
Research PDF
    │
    ▼
PDF → Markdown → Cleaning / Normalization
    │
    ▼
Semantic Chunking + Metadata
    │
    ├──────────────────────┬────────────────────────┐
    ▼                      ▼                        ▼
Embeddings               BM25                 Entity / Relation
    │                      │                     Extraction
    ▼                      ▼                        ▼
Chroma Vector DB       Lexical Index              Neo4j
    │                      │                        │
    └──────────────┬───────┘                        │
                   ▼                                │
             RRF Fusion                             │
                   │                                │
                   ▼                                │
            Cross-Encoder                           │
              Reranking                             │
                   │                                │
                   └──────────────┬─────────────────┘
                                  ▼
                         6-Step Generation
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
          Guardrail          Query Rewrite         Retrieve
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ▼
                            Scope Check
                                  │
                                  ▼
                               Draft
                                  │
                                  ▼
                               Verify
                                  │
                                  ▼
                         Grounded Answer
```

---

## 🎯 Why This Project?

A basic RAG pipeline often looks like:

```text
PDF → Embeddings → Vector DB → LLM
```

That approach can retrieve semantically similar text, but it may miss:

- Exact medical terminology
- Keyword-level matches
- Explicit relationships between entities
- The most relevant passage among many candidates
- Evidence consistency during generation

This project addresses those limitations with a **multi-stage retrieval and verification architecture**:

| Layer | Purpose |
|---|---|
| **Semantic Search** | Understand meaning and retrieve conceptually relevant chunks |
| **BM25** | Capture exact terms, abbreviations, and lexical matches |
| **RRF Fusion** | Combine semantic and keyword rankings |
| **Cross-Encoder** | Re-rank candidates using query-document relevance |
| **Neo4j** | Retrieve explicit entity relationships |
| **Scope Check** | Ensure evidence is relevant to the source domain |
| **Grounded Drafting** | Generate from retrieved evidence |
| **Verification** | Check claims against available evidence |

---

# 🔬 Core Pipeline

## 1. Data Ingestion

The system starts from the hepatology research paper.

```text
PDF
 ↓
pdf_to_markdown.py
 ↓
Structured Markdown
```

The conversion stage preserves useful document structure such as headings, sections, and source information.

---

## 2. Preprocessing & Chunking

The Markdown document is split into retrieval-friendly chunks.

Default configuration:

```text
Chunk size      = 512 tokens
Chunk overlap   = 80 tokens
```

Each chunk can carry metadata such as:

- Section
- Page range
- DOI
- Source
- Chunk ID

This metadata is later used for traceability and citations.

---

## 3. Dual Indexing

The corpus is indexed in two complementary ways.

### A. Semantic / Dense Retrieval

Embedding model:

```text
BAAI/bge-large-en-v1.5
```

Pipeline:

```text
Chunk
  ↓
Embedding Model
  ↓
Vector
  ↓
Chroma
```

Semantic retrieval is useful when the query and the source text use different wording but express similar concepts.

### B. Lexical Retrieval

The same chunks are indexed using **BM25**.

```text
Query
  ↓
BM25
  ↓
Keyword-based ranking
```

BM25 is particularly useful for:

- Medical abbreviations
- Exact disease names
- Drug names
- Gene names
- Technical terminology

---

# 🔀 4. Hybrid Retrieval

The system does not choose between semantic search and BM25.

It combines them.

```text
                 User Query
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Semantic Search           BM25 Search
      (Chroma)                (Lexical)
          │                     │
          └──────────┬──────────┘
                     ▼
                RRF Fusion
                     │
                     ▼
              Candidate Pool
                     │
                     ▼
             Cross-Encoder
                Reranker
                     │
                     ▼
               Top Results
```

### Reciprocal Rank Fusion (RRF)

RRF combines the rankings produced by different retrieval methods instead of relying on a single score space.

This makes the retrieval stage more robust because a document can be highly relevant semantically, lexically, or through both signals.

---

# 🧠 5. Knowledge Graph
<img width="1064" height="625" alt="visualisation" src="https://github.com/user-attachments/assets/4352075f-8dda-4d13-8f1c-9ad62a79d6bb" />



The system also extracts entities and relationships from the source document.

```text
Chunks
  ↓
LLM-based Entity / Relation Extraction
  ↓
graph_triples.json
  ↓
Neo4j
```

Conceptually, the graph can represent relationships such as:

```text
Disease ── associated_with ── Risk Factor
Disease ── caused_by ──────── Virus
Disease ── increases_risk ─── Cancer
Drug ──── used_for ────────── Disease
```

Graph retrieval adds a structured relationship layer that complements vector and lexical retrieval.

---

# 🤖 6. Six-Step Grounded Generation Chain

The answer-generation pipeline is intentionally more than:

```text
Retrieve → Ask LLM → Answer
```

It uses six stages:

### 1. Guardrail

Reject or limit requests outside the intended research-Q&A scope.

### 2. Query Rewrite

Normalize or expand the user's question when useful, including medical abbreviations and terminology.

### 3. Retrieve

Run the hybrid retrieval pipeline:

```text
Semantic + BM25
       ↓
      RRF
       ↓
   Reranking
       ↓
 Top-k Context
```

Graph facts can also be incorporated as additional structured evidence.

### 4. Scope Check

Determine whether the retrieved evidence is sufficiently relevant to the paper and the question.

### 5. Draft

Generate an answer using the retrieved evidence rather than relying only on the model's internal knowledge.

### 6. Verify

Check generated claims against the available evidence and avoid presenting unsupported content as fact.

---

# 🧩 7. Production API

The backend is implemented with **FastAPI**.

Main endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Production dashboard |
| `GET` | `/health` | Health and dependency status |
| `POST` | `/search` | Hybrid retrieval |
| `POST` | `/ask` | Grounded Q&A |
| `GET` | `/graph/{entity}` | Knowledge-graph lookup |

### `/search`

Retrieval-only endpoint:

```text
Semantic Search
      +
BM25
      ↓
RRF
      ↓
Reranking
      ↓
Top-k chunks
```

No generation is performed here.

### `/ask`

Full answer pipeline:

```text
Question
   ↓
Guardrail
   ↓
Query Rewrite
   ↓
Hybrid Retrieval
   ↓
Graph Context
   ↓
Scope Check
   ↓
Draft
   ↓
Verify
   ↓
Answer
```

---

# 📊 8. Evaluation

The project includes a benchmark dataset with **27 questions**.

Evaluation metrics include:

- Hit Rate@K
- Precision@K
- MRR
- Latency
- Per-question failure analysis

### Retrieval Results

| Metric | RRF Only | RRF + Reranking |
|---|---:|---:|
| Hit Rate@5 | 92.6% | **100.0%** |
| Avg Precision@5 | 54.8% | **57.0%** |
| MRR | 85.9% | **90.1%** |
| Avg Latency | 0.147s | 1.038s |

### Interpretation

Reranking substantially improves retrieval quality:

```text
Hit Rate@5
92.6%  →  100%

MRR
85.9%  →  90.1%
```

The trade-off is latency:

```text
0.147s  →  1.038s
```

This demonstrates an important production principle:

> **Better retrieval quality usually comes with additional compute and latency.**

---

# 🖥️ 9. Interactive Dashboard

The project includes a standalone web dashboard for interacting with the system.

The dashboard provides:

- Live health status
- Quick search
- Deep search
- Full-answer mode
- Graph exploration
- Evaluation information
- Pipeline visualization

The FastAPI application also serves the dashboard at:

```text
GET /
```

---

# ☁️ 10. Production Deployment

The system is designed for low-RAM deployment.

## Production architecture

```text
                         ┌─────────────────────┐
                         │       Railway       │
                         │      FastAPI        │
                         └─────────┬───────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
      Hugging Face API          Jina API               Groq
       Query Embeddings        Reranking             Generation
              │                    │                     │
              └────────────────────┼─────────────────────┘
                                   ▼
                         Hybrid RAG Pipeline
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                       Chroma             BM25
                          │                 │
                          └────────┬────────┘
                                   ▼
                                  RRF
                                   │
                                   ▼
                                Top-k
                                   │
                                   ▼
                                 Neo4j
```

### Why remote inference?

Large ML models can consume significant RAM.

Instead of loading every model inside a small Railway instance:

```text
Railway
  ├── FastAPI
  ├── Chroma
  ├── BM25
  └── Application logic

Remote APIs
  ├── Hugging Face → embeddings
  ├── Jina → reranking
  └── Groq → generation
```

This keeps the application lightweight while preserving the retrieval architecture.

### Railway configuration

Production mode uses:

```env
ENVIRONMENT=production
USE_HF_INFERENCE_API=true
USE_JINA_RERANKER_API=true

GROQ_API_KEY=...
HF_API_TOKEN=...
JINA_API_KEY=...

NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
```

> Never commit `.env` or API keys to GitHub.

---

# 🚀 11. Getting Started

## Clone

```bash
git clone https://github.com/abdo7820/rag.git
cd rag
```

## Create environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install local dependencies

```bash
pip install -r requirements-local.txt
```

## Configure environment

Create:

```text
.env
```

Example:

```env
GROQ_API_KEY=your_groq_key

ENVIRONMENT=development

USE_HF_INFERENCE_API=false
USE_JINA_RERANKER_API=false

NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

---

# 📚 12. Build the Knowledge Base

Place the source paper at:

```text
data/s41392-024-02072-z.pdf
```

Run the pipeline in order:

```bash
python rag/pdf_to_markdown.py
python rag/chunker.py
python rag/embed.py
python rag/store.py
python rag/bm25_index.py
```

Optional Knowledge Graph:

```bash
python models/extract_graph.py
python models/load_graph.py
```

---

# ▶️ 13. Run the API

```bash
uvicorn app:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🧪 14. Run Evaluation

```bash
python eval/evaluate_retrieval.py
```

Results are written to:

```text
eval/results.json
```

The evaluation compares:

```text
RRF
vs.
RRF + Cross-Encoder Reranking
```

---

# 📁 15. Project Structure

```text
rag/
│
├── app.py
├── config.py
├── dashboard.html
├── Dockerfile
├── Procfile
├── railway.json
│
├── data/
│   ├── s41392-024-02072-z.pdf
│   ├── chunks.json
│   ├── embeddings.npy
│   ├── embeddings_meta.json
│   ├── bm25_index.pkl
│   └── graph_triples.json
│
├── rag/
│   ├── pdf_to_markdown.py
│   ├── chunker.py
│   ├── embed.py
│   ├── store.py
│   ├── bm25_index.py
│   ├── test_search.py
│   └── generate.py
│
├── models/
│   ├── extract_graph.py
│   ├── load_graph.py
│   └── query_graph.py
│
├── vectorDB/
│   └── Chroma database
│
└── eval/
    ├── qa_dataset.json
    └── evaluate_retrieval.py
```

---

# ⚙️ 16. Configuration

All major parameters are centralized in:

```text
config.py
```

Important settings include:

```python
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 80

CANDIDATE_K = 30
RRF_K = 60
RRF_TOP_K = 8
FINAL_TOP_K = 5

RETRIEVAL_TOP_K = 5
GRAPH_FACTS_LIMIT = 5
```

This makes the system easier to tune and reproduce.

---

# 🔐 17. Security & Safety

The project follows several production-oriented practices:

- Secrets are loaded from environment variables.
- API keys are not hard-coded.
- Production configuration validates required credentials.
- The API exposes a health endpoint.
- The application uses a single worker for low-RAM deployment.
- The guardrail prevents some out-of-scope requests.

### Important limitation

This is a **research-paper Q&A system**, not a clinical decision-support system.

The current guardrail is a keyword-based heuristic and is not sufficient for real-world medical deployment.

The system should **not** be used as a substitute for a qualified medical professional.

---

# ⚖️ 18. Design Trade-offs

### Dense Retrieval

**Pros**
- Understands semantic similarity
- Handles paraphrasing

**Cons**
- Can miss exact terminology

### BM25

**Pros**
- Excellent lexical matching
- Useful for medical terms and abbreviations

**Cons**
- Less semantic understanding

### RRF

**Pros**
- Combines multiple ranking signals
- Does not require scores from different systems to be directly comparable

**Cons**
- Adds another retrieval stage

### Cross-Encoder

**Pros**
- Strong relevance ranking

**Cons**
- Higher latency

### Knowledge Graph

**Pros**
- Explicit relationships
- Useful for entity-centric questions

**Cons**
- Requires extraction and graph maintenance

---

# 🧠 19. What Makes This RAG Different?

The main architectural idea is:

```text
                  ┌────────────────────┐
                  │     User Query     │
                  └─────────┬──────────┘
                            │
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
        Semantic          BM25             Neo4j
        Retrieval        Retrieval          Graph
            │               │                │
            └───────┬───────┘                │
                    ▼                        │
                 RRF Fusion                  │
                    │                        │
                    ▼                        │
              Cross-Encoder                 │
                Reranking                   │
                    │                        │
                    └───────────┬────────────┘
                                ▼
                         Grounded Context
                                │
                                ▼
                       6-Step Generation
                                │
                                ▼
                     Verified Answer
```

Instead of trusting one retrieval method, the system uses **multiple independent evidence signals** and then verifies the generated response against the retrieved context.

---

# 📌 20. Key Technologies

| Category | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| LLM | Groq |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Vector DB | Chroma |
| Keyword Search | BM25 |
| Fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder / Jina API |
| Knowledge Graph | Neo4j |
| Graph Extraction | Llama 3.1 8B |
| Deployment | Railway |
| Remote Inference | Hugging Face Inference API |
| Containerization | Docker |
| Evaluation | Hit Rate, Precision, MRR, Latency |

---

# 📈 21. Future Improvements

Potential next steps:

- Multi-document knowledge base
- Multilingual retrieval
- Better medical-domain reranker
- LLM-based safety classifier
- Citation-level factuality evaluation
- Answer-level evaluation with LLM judges
- Query decomposition for complex questions
- Agentic retrieval
- Temporal knowledge graph
- Multimodal PDF understanding
- Automated ingestion of new research papers
- Authentication and rate limiting
- Observability and production tracing

---

# 📖 22. Source Paper

**Gan et al.**  
*Liver diseases: epidemiology, causes, trends and predictions.*

Signal Transduction and Targeted Therapy.

DOI:

```text
10.1038/s41392-024-02072-z
```

---

# 👨‍💻 Author

**Abdelrahman Mohamed Yousry**

Data Scientist | Machine Learning Engineer | Deep Learning Enthusiast

- GitHub: [@abdo7820](https://github.com/abdo7820)
- Repository: [Liver Diseases RAG](https://github.com/abdo7820/rag)

---

## ⭐ If you find this project useful

Give the repository a ⭐ and feel free to explore, improve, or extend the architecture.

---

<p align="center">
  <b>Built with Python • FastAPI • Chroma • BM25 • Neo4j • Groq • Hugging Face • Jina</b>
</p>
