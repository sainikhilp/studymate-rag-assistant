# RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions grounded in local PDF documents (textbook + syllabus). Uses OpenAI for embeddings and chat completion, FAISS as the local vector store.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# Edit .env and replace sk-... with your real OpenAI API key
```

## Place your PDFs

Drop your PDF files into the `data/` directory:

```
data/
  course_textbook.pdf
  course_syllabus.pdf
```

## Build the index

```bash
python -m src.build_index
```

The script will:
1. Extract text from every `.pdf` in `data/`
2. Chunk text recursively with 800-token chunks and 150-token overlap
3. Embed all chunks via OpenAI `text-embedding-3-small`
4. Save `index/faiss.index` and `index/metadata.pkl`

**Rebuild** (if you add new PDFs or want to regenerate):

```bash
python -m src.build_index --rebuild
```

## Chat

```bash
python -m src.chat
```

### Chat commands

| Command | Action |
|---------|--------|
| `exit` / `quit` | Exit the chatbot |
| `clear` | Reset conversation history |
| `sources` | Show sources from the last answer |
| `help` | Show command list |

## Cost estimate

Embedding cost for `text-embedding-3-small` is **$0.02 per 1M tokens**.

| Document size | Approx. tokens | Approx. cost |
|---------------|----------------|--------------|
| 100-page PDF  | ~50,000         | ~$0.001      |
| 500-page PDF  | ~250,000        | ~$0.005      |
| 1,000-page PDF | ~500,000       | ~$0.010      |

A typical textbook + syllabus combination costs **$0.01–$0.05 total** for the one-time embedding step.

## Project structure

```
data/                  PDF source documents
index/                 FAISS index + metadata (generated)
src/
  config.py            Central config (paths, models, params)
  pdf_loader.py        PDF text extraction (pypdf)
  chunker.py           Recursive character chunking (tiktoken)
  embedder.py          Batched OpenAI embeddings with retry
  build_index.py       Build pipeline: load → chunk → embed → save
  rag.py               Retrieval + prompt + chat completion
  chat.py              Interactive CLI loop
notebooks/
  explore.ipynb        Load index and run example queries
.env.example           API key template
requirements.txt       Python dependencies
```
