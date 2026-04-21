from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import random
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# ── 1. Load PDFs ──────────────────────────────────────────────────────────────
pdf_files = [
    "../data/course_syllabus.pdf",
    "../data/course_textbook.pdf",
]

docs = []
for path in tqdm(pdf_files, desc="📄 Loading PDFs"):
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

print(f"✅ Loaded {len(docs)} pages total")

# ── 2. Chunk ──────────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# Use syllabus chunks + a sample of textbook chunks
syllabus_chunks = [c for c in chunks if "syllabus" in c.metadata.get("source", "")]
textbook_chunks = [c for c in chunks if "textbook" in c.metadata.get("source", "")]

selected = syllabus_chunks + random.sample(textbook_chunks, min(50, len(textbook_chunks)))
random.shuffle(selected)
print(f"✅ Using {len(selected)} chunks ({len(syllabus_chunks)} syllabus + up to 50 textbook)")

# ── 3. Generate Q&A pairs via OpenAI ─────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert at creating evaluation datasets for RAG systems.
Given a text chunk, generate a question and ground truth answer that:
- Can be answered from the chunk alone
- Is specific and unambiguous
- Varies in type: factual, reasoning, or multi-part

Respond ONLY with valid JSON in this exact format:
{"question": "...", "ground_truth": "..."}"""

def generate_qa(chunk_text: str) -> dict | None:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Chunk:\n{chunk_text}"},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        return None

# ── 4. Build dataset ──────────────────────────────────────────────────────────
TARGET = 100  # number of Q&A pairs — increase once confirmed working
records = []

for chunk in tqdm(selected[:TARGET], desc="🧪 Generating Q&A pairs"):
    qa = generate_qa(chunk.page_content)
    if qa:
        records.append({
            "question":     qa["question"],
            "ground_truth": qa["ground_truth"],
            "contexts":     [chunk.page_content],
            "source":       chunk.metadata.get("source", ""),
            "page":         chunk.metadata.get("page", ""),
        })

print(f"✅ Generated {len(records)} Q&A pairs")

# ── 5. Save ───────────────────────────────────────────────────────────────────
os.makedirs("evaluation", exist_ok=True)
df = pd.DataFrame(records)
df.to_csv("evaluation/testset.csv", index=False)
print("✅ Saved to evaluation/testset.csv")
print(df[["question", "ground_truth", "source", "page"]].head(10).to_string())