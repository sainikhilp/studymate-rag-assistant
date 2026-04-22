
# studymate-rag-chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about course material — the syllabus and textbook — using OpenAI embeddings, FAISS vector search, and GPT-4o mini. Includes a Gradio web UI, a terminal REPL, and a full evaluation pipeline comparing three OpenAI models.

---

## Demo

The chatbot ingests your course PDFs, splits them into overlapping chunks, embeds them with `text-embedding-3-small`, and stores them in a FAISS index. At query time it retrieves the most relevant chunks and passes them to GPT-4o mini with a strict grounding prompt no hallucination from outside knowledge.

---

## Features

- **Grounded answers** — responds only from retrieved context; says "I don't know" when unsupported
- **Source citations** — every answer includes `[filename p.N]` references with similarity scores
- **Multi-turn conversation** — full chat history passed to the LLM for natural follow-up questions
- **Small-doc guarantee** — all 12 syllabus chunks are always included so syllabus questions are never missed
- **Web UI** — Gradio interface with live source panel, raw-chunk toggle, and example questions
- **Terminal REPL** — lightweight CLI for quick queries
- **Comprehensive evaluation** — automated LLM-judge metrics + human scoring across 3 models

---

## Architecture

```
PDFs (syllabus + textbook)
        │
        ▼
   pdf_loader.py         ← extract text, page-by-page
        │
        ▼
    chunker.py           ← recursive split, 800-token chunks, 150-token overlap
        │
        ▼
    embedder.py          ← OpenAI text-embedding-3-small (1536-dim), L2-normalized
        │
        ▼
  build_index.py         ← FAISS IndexFlatIP  →  index/faiss.index + metadata.pkl

At query time:
  User question  →  embed  →  FAISS search  →  top-k chunks  →  GPT-4o mini  →  Answer + sources
```

---

## Project Structure

```
chatbot/
├── app.py                      # Gradio web UI (main entry point)
├── requirements.txt            # pip dependencies
├── pyproject.toml              # uv project config
│
├── src/
│   ├── config.py               # Central settings (models, paths, chunk params)
│   ├── pdf_loader.py           # Extract text from PDFs
│   ├── chunker.py              # Recursive token-aware chunking
│   ├── embedder.py             # OpenAI embeddings with retry + batching
│   ├── build_index.py          # Orchestrate index creation
│   ├── rag.py                  # Retrieval + answer generation
│   └── chat.py                 # Terminal REPL
│
├── data/
│   ├── course_syllabus.pdf     # 12 pages
│   └── course_textbook.pdf     # 626 pages
│
├── index/
│   ├── faiss.index             # 895 vectors × 1536 dims (auto-generated)
│   └── metadata.pkl            # Chunk metadata (auto-generated)
│
├── notebooks/
│   └── explore.ipynb           # Interactive exploration & debugging
│
└── evaluation/
    ├── generate_test_data.py   # Generate 100 Q&A test pairs from PDFs
    ├── evaluate_models.py      # Compare gpt-4o-mini, gpt-4o, o1-mini
    ├── analyze_results.py      # Aggregate results + generate plots
    ├── human_scorer.html       # Interactive manual scoring UI
    ├── testset.csv             # 100 test questions
    └── plots/                  # 6 evaluation visualizations
```

---

## Quickstart

### Prerequisites

- Python 3.12+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Installation

```bash
git clone https://github.com/your-username/chatbot.git
cd chatbot

# With uv (recommended)
pip install uv
uv sync

# Or with pip
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
OPEN_AI_KEY=sk-proj-your-key-here
```

### Build the index

Run once after cloning (or after updating PDFs):

```bash
python -m src.build_index
# Force rebuild:  python -m src.build_index --rebuild
```

This embeds ~895 chunks and costs roughly **$0.013** in OpenAI embedding credits.

### Run

**Web UI** (recommended):
```bash
python app.py
# Opens at http://127.0.0.1:7860
```

**Terminal REPL:**
```bash
python -m src.chat
```

**Jupyter notebook:**
```bash
jupyter notebook notebooks/explore.ipynb
```

---

## Configuration

All tuneable parameters live in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat completion model |
| `CHUNK_SIZE` | 800 tokens | Max tokens per chunk |
| `CHUNK_OVERLAP` | 150 tokens | Overlap between consecutive chunks |
| `MIN_CHUNK_TOKENS` | 30 | Skip chunks below this size |
| `TOP_K` | 6 | Chunks retrieved per query |
| `TEMPERATURE` | 0.2 | LLM generation temperature |

---

## Evaluation

The evaluation pipeline compares three OpenAI models — **gpt-4o-mini**, **gpt-4o**, and **o1-mini** — across 100 test questions using both automated LLM-judge metrics and manual human scoring.

### Human Scores

> Average human rating (1–5 scale) across 20 manually scored answers per model.

![Human scores by model](evaluation/plots/01_human_scores.png)

### Automated Metrics

> Faithfulness, answer relevancy, context precision, and context recall — scored by an LLM judge.

![Auto metrics bar chart](evaluation/plots/02_auto_metrics_bar.png)

![Auto metrics radar chart](evaluation/plots/03_auto_metrics_radar.png)

### Cost vs. Quality

> Quality score vs. cost per answer — the key trade-off when choosing a model.

![Cost vs quality](evaluation/plots/04_cost_vs_quality.png)

### Response Time

> Median and P95 latency per model.

![Response time](evaluation/plots/05_response_time.png)

### Human Score Distribution

> Distribution of per-question human scores for each model.

![Human score distribution](evaluation/plots/06_human_score_dist.png)

### Key Findings

- **gpt-4o-mini** delivers the best cost-quality trade-off at ~$0.002/turn — chosen as the default
- **gpt-4o** scores marginally higher on human ratings but costs ~10× more per answer
- **o1-mini** excels on reasoning-heavy questions but is slower and more expensive
- All models score high on faithfulness (>0.90) thanks to the strict grounding prompt

---

## How RAG Works

1. **Indexing (one-time):** PDFs are loaded page-by-page, recursively split into 800-token overlapping chunks, embedded with `text-embedding-3-small`, and stored in a FAISS flat index.

2. **Retrieval (per query):** The user's question is embedded with the same model. FAISS performs exact inner-product search (equivalent to cosine similarity on L2-normalized vectors). The top-6 textbook chunks are returned, plus all 12 syllabus chunks unconditionally.

3. **Generation:** Retrieved chunks + chat history are assembled into a prompt. GPT-4o mini is instructed to answer only from the provided context and cite sources as `[filename p.N]`.

---
# Stydymate-rag-assi

## Tech Stack

| Component | Library |
|-----------|---------|
| PDF parsing | `pypdf` |
| Tokenization | `tiktoken` (cl100k_base) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector search | `faiss-cpu` (IndexFlatIP) |
| LLM | OpenAI `gpt-4o-mini` |
| Web UI | `gradio` |
| Package manager | `uv` |

---

## Cost Reference

| Operation | Cost |
|-----------|------|
| Build index (895 chunks, one-time) | ~$0.013 |
| Chat turn (gpt-4o-mini, ~6 chunks context) | ~$0.002 |
| Full evaluation run (100 questions × 3 models) | ~$2–5 |

---

## License

MIT
