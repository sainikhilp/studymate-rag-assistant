"""Central configuration: paths, model names, and chunking parameters."""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
INDEX_DIR: Path = ROOT_DIR / "index"
FAISS_INDEX_PATH: Path = INDEX_DIR / "faiss.index"
METADATA_PATH: Path = INDEX_DIR / "metadata.pkl"

# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIM: int = 1536
CHAT_MODEL: str = "gpt-4o-mini"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS: int = 800
CHUNK_OVERLAP_TOKENS: int = 150
MIN_CHUNK_TOKENS: int = 30
TIKTOKEN_ENCODING: str = "cl100k_base"

# ── Retrieval ──────────────────────────────────────────────────────────────────
# Top-k chunks from large documents (e.g. the textbook)
TOP_K: int = 6
# If a source document has <= this many chunks, include ALL of its chunks on
# every query. This prevents small docs (e.g. a 12-chunk syllabus) from having
# their key facts buried below the TOP_K cutoff due to query-phrasing variance.
SMALL_DOC_CHUNK_THRESHOLD: int = 20
TEMPERATURE: float = 0.2

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBED_BATCH_SIZE: int = 100

# ── Cost estimate ──────────────────────────────────────────────────────────────
EMBED_COST_PER_MILLION_TOKENS: float = 0.02
