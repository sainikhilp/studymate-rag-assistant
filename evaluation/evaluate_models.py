"""
RAG Model Comparison Evaluator
Compares 3 OpenAI models on your RAG setup across:
  - Faithfulness, Answer Relevancy, Context Precision, Context Recall
    → LLM judge (gpt-4o) runs on ALL 100 samples per model
  - Cost, Response Time, Token Usage
  - Human evaluation scores (20 samples per model, flagged via `human_eval_sample=True`)
    → Load evaluation/model_comparison_raw.csv into human_eval.html to score

Usage:
  python evaluation/evaluate_models.py           # uses up to 100 samples
  python evaluation/evaluate_models.py --samples 50
  python evaluation/evaluate_models.py --human 10  # override human sample count
"""

import sys, os, time, json, ast, argparse, random
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

from src.rag import load_index, answer  # your RAG internals
from src import config

# ── Models to compare ─────────────────────────────────────────────────────────
MODELS = [
    {
        "id":    "gpt-4o-mini",
        "label": "GPT-4o Mini  (fast / cheap)",
        "tier":  "low",
        "cost_input_per_1k":  0.00015,   # $ per 1K input tokens
        "cost_output_per_1k": 0.00060,
    },
    {
        "id":    "gpt-4o",
        "label": "GPT-4o  (balanced)",
        "tier":  "mid",
        "cost_input_per_1k":  0.00250,
        "cost_output_per_1k": 0.01000,
    },
    {
        "id":    "o1-mini",
        "label": "o1-mini  (reasoning / best)",
        "tier":  "high",
        "cost_input_per_1k":  0.00110,
        "cost_output_per_1k": 0.00440,
    },
]

# ── Auto-evaluation judge prompt ──────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
You will be given a question, context chunks retrieved from a knowledge base, and an answer.
Score the answer on four dimensions from 0.0 to 1.0:

1. faithfulness      — Is every claim in the answer supported by the provided context?
2. answer_relevancy  — Does the answer directly address the question asked?
3. context_precision — Do the retrieved context chunks actually contain info needed to answer?
4. context_recall    — Does the answer cover all important information from the context?

You MUST respond with ONLY a raw JSON object. No markdown, no backticks, no explanation.
Example output: {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.85}"""

JUDGE_NULL = {"faithfulness": None, "answer_relevancy": None,
              "context_precision": None, "context_recall": None}

def _parse_judge_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON robustly."""
    raw = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # Find the first { ... } block in case there's any stray text
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in: {raw!r}")
    return json.loads(raw[start:end])

def judge_answer(client: OpenAI, question: str, context: list[str], answer_text: str,
                 retries: int = 3) -> dict:
    ctx_str  = "\n---\n".join(context[:6])
    user_msg = (
        f"Question: {question}\n\n"
        f"Contexts:\n{ctx_str}\n\n"
        f"Answer:\n{answer_text}\n\n"
        f"Respond with ONLY the JSON object. No other text."
    )
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                max_tokens=150,
                response_format={"type": "json_object"},  # force JSON mode
            )
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                raise ValueError("Empty response from judge")
            return _parse_judge_response(raw)
        except Exception as e:
            if attempt == retries:
                print(f"  ⚠️  Judge failed after {retries} attempts: {e}")
                return JUDGE_NULL
            time.sleep(1 * attempt)  # brief backoff before retry


def run_rag_with_model(question: str, index, metadata, model_id: str) -> dict:
    """Run a single RAG query, override the chat model, and capture timing + tokens."""
    # Temporarily swap config model
    original_model = config.CHAT_MODEL
    config.CHAT_MODEL = model_id

    start = time.perf_counter()
    result = answer(question, index, metadata, history=[])
    elapsed = time.perf_counter() - start

    config.CHAT_MODEL = original_model

    return {
        "answer":       result["answer"],
        "sources":      result["sources"],
        "response_time": round(elapsed, 2),
        # token counts — add these to your answer() return dict if not present
        "input_tokens":  result.get("input_tokens",  0),
        "output_tokens": result.get("output_tokens", 0),
    }


def compute_cost(model: dict, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  / 1000 * model["cost_input_per_1k"] +
        output_tokens / 1000 * model["cost_output_per_1k"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples for LLM judge (default: 100)")
    parser.add_argument("--human", type=int, default=20,
                        help="Number of samples to flag for human eval per model (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for human sample selection (default: 42)")
    args = parser.parse_args()

    LLM_SAMPLES   = args.samples   # 100 — judged automatically
    HUMAN_SAMPLES = args.human     # 20  — flagged for human review

    # ── Load testset ──────────────────────────────────────────────────────────
    testset_path = "evaluation/testset.csv"
    if not os.path.exists(testset_path):
        print(f"❌ {testset_path} not found. Run generate_test_data.py first.")
        sys.exit(1)

    df = pd.read_csv(testset_path)
    df = df[df["ground_truth"].notna() & (df["ground_truth"].str.strip() != "")]

    if len(df) < LLM_SAMPLES:
        print(f"⚠️  Only {len(df)} samples available (requested {LLM_SAMPLES}). Using all.")
        LLM_SAMPLES = len(df)

    df = df.head(LLM_SAMPLES).reset_index(drop=True)

    # Pick 20 random positions (0-based) to flag for human eval — same across all models
    random.seed(args.seed)
    human_indices = set(random.sample(range(len(df)), min(HUMAN_SAMPLES, len(df))))
    print(f"✅ LLM judge  : {LLM_SAMPLES} samples per model")
    print(f"✅ Human eval : {len(human_indices)} samples per model (flagged in CSV)")
    print(f"✅ Models     : {len(MODELS)}\n")

    client          = OpenAI()
    index, metadata = load_index()
    all_rows        = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  Model: {model['label']}")
        print(f"{'='*60}")

        for pos, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"  Querying {model['id']}")):
            question        = row["question"]
            ground_truth    = row["ground_truth"]
            is_human_sample = pos in human_indices  # use pos (0-based), not df index

            # Parse context from CSV (stored as string repr of list)
            try:
                raw_ctx = row.get("contexts", "[]")
                csv_ctx = ast.literal_eval(raw_ctx) if isinstance(raw_ctx, str) else raw_ctx
            except Exception:
                csv_ctx = [str(raw_ctx)]

            # Run RAG
            try:
                rag_result = run_rag_with_model(question, index, metadata, model["id"])
            except Exception as e:
                print(f"\n  ⚠️  RAG error on sample {pos}: {e}")
                continue

            answer_text   = rag_result["answer"]
            response_time = rag_result["response_time"]
            input_tok     = rag_result["input_tokens"]
            output_tok    = rag_result["output_tokens"]
            cost          = compute_cost(model, input_tok, output_tok)

            # Build context from RAG sources (prefer live retrieval over CSV)
            live_ctx = [s["document"] for s in rag_result["sources"] if "document" in s]
            context  = live_ctx if live_ctx else csv_ctx

            # Auto-evaluate with judge LLM
            scores = judge_answer(client, question, context, answer_text)

            all_rows.append({
                "model_id":           model["id"],
                "model_label":        model["label"],
                "model_tier":         model["tier"],
                "sample_index":       pos,
                "question":           question,
                "ground_truth":       ground_truth,
                "answer":             answer_text,
                "response_time_s":    response_time,
                "input_tokens":       input_tok,
                "output_tokens":      output_tok,
                "cost_usd":           round(cost, 6),
                "faithfulness":       scores.get("faithfulness"),
                "answer_relevancy":   scores.get("answer_relevancy"),
                "context_precision":  scores.get("context_precision"),
                "context_recall":     scores.get("context_recall"),
                # human eval — only flagged samples shown in the UI
                "human_eval_sample":  is_human_sample,
                "human_score":        None,
                "human_notes":        None,
            })

    # ── Save raw results ──────────────────────────────────────────────────────
    os.makedirs("evaluation", exist_ok=True)
    results_df = pd.DataFrame(all_rows)
    raw_path   = "evaluation/model_comparison_raw.csv"
    results_df.to_csv(raw_path, index=False)
    print(f"\n✅ Raw results saved → {raw_path}")
    print(f"   Total rows : {len(results_df)}  ({LLM_SAMPLES} samples × {len(MODELS)} models)")
    print(f"   Human eval : {len(human_indices)} flagged samples per model  "
          f"(human_eval_sample=True in CSV)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n📊 Summary — LLM judge scores (all 100 samples):")
    print("-" * 95)
    fmt = "{:<30} {:>10} {:>10} {:>12} {:>10} {:>10} {:>12} {:>8}"
    print(fmt.format("Model", "Faithf.", "Relev.", "Ctx Prec.", "Ctx Rec.", "Avg RT(s)", "Total Cost", "Out Tok"))
    print("-" * 95)

    for model in MODELS:
        mdf = results_df[results_df["model_id"] == model["id"]]
        print(fmt.format(
            model["label"][:30],
            f"{mdf['faithfulness'].mean():.3f}",
            f"{mdf['answer_relevancy'].mean():.3f}",
            f"{mdf['context_precision'].mean():.3f}",
            f"{mdf['context_recall'].mean():.3f}",
            f"{mdf['response_time_s'].mean():.2f}s",
            f"${mdf['cost_usd'].sum():.4f}",
            f"{int(mdf['output_tokens'].mean())}",
        ))
    print("-" * 95)

    print(f"\n👤 Human eval next steps:")
    print(f"   1. Open evaluation/human_eval.html in your browser")
    print(f"   2. Load evaluation/model_comparison_raw.csv")
    print(f"   3. The UI will show only the {len(human_indices)} flagged samples per model")
    print(f"   4. Score each → export → save as evaluation/model_comparison_scored.csv\n")


if __name__ == "__main__":
    main()