"""Gradio demo UI for the RAG chatbot."""

import gradio as gr
from src.rag import load_index, answer as rag_answer

# ── Load index once at startup ─────────────────────────────────────────────────
print("Loading index...")
_index, _metadata = load_index()
print("Index ready.")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_sources(sources: list[dict]) -> str:
    """Render retrieved sources as a compact markdown list."""
    if not sources:
        return "_No sources retrieved._"
    rows = ["| # | Document | Page | Score |", "|---|-----------|------|-------|"]
    for i, s in enumerate(sources, 1):
        doc = s["source_file"].replace("course_", "").replace(".pdf", "").capitalize()
        rows.append(f"| {i} | {doc} | p. {s['page_number']} | {s['score']:.3f} |")
    return "\n".join(rows)


def _format_chunk_preview(sources: list[dict]) -> str:
    """Render the raw retrieved chunk text for debugging."""
    if not sources:
        return "_No chunks._"
    parts = []
    for i, s in enumerate(sources, 1):
        doc = s["source_file"].replace("course_", "").replace(".pdf", "").capitalize()
        preview = s["document"][:400].replace("\n", " ").strip()
        parts.append(
            f"**[{i}] {doc} — p. {s['page_number']} (score: {s['score']:.3f})**\n\n"
            f"> {preview}…"
        )
    return "\n\n---\n\n".join(parts)


# ── Core chat function ─────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list[dict],
    show_chunks: bool,
) -> tuple[list[dict], str, str]:
    """Process one user turn; returns (updated_history, sources_md, cleared_input)."""
    if not user_message.strip():
        return history, _format_sources([]), ""

    openai_history = [{"role": m["role"], "content": m["content"]} for m in history]
    result = rag_answer(user_message, _index, _metadata, history=openai_history)
    reply = result["answer"]
    sources = result["sources"]

    updated = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    sources_md = _format_chunk_preview(sources) if show_chunks else _format_sources(sources)
    return updated, sources_md, ""


def clear_all() -> tuple[list, str, str]:
    return [], "_Sources will appear here after your first question._", ""


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
h1 { font-size: 1.6rem !important; margin-bottom: 0.2rem !important; }
.subtitle { color: #64748b; font-size: 0.92rem; margin-bottom: 1rem; }
#chatbot { height: 500px; }
#sources-panel { height: 500px; overflow-y: auto; font-size: 0.85em; }
#send-btn { min-width: 80px; }
footer { display: none !important; }
"""

# ── UI layout ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Course RAG Chatbot") as demo:

    gr.HTML("""
        <h1>Course RAG Chatbot</h1>
        <p class="subtitle">
            Ask anything about the <strong>course syllabus</strong> or
            <strong>textbook</strong>. Every answer is grounded in your documents.
        </p>
    """)

    with gr.Row():

        # ── Left: chat ──────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                elem_id="chatbot",
                height=500,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=rag"),
                show_label=False,
                placeholder="Ask a question to get started...",
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask about the course or textbook...",
                    show_label=False,
                    scale=5,
                    lines=1,
                    max_lines=3,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

            with gr.Row():
                clear_btn = gr.Button("Clear conversation", variant="stop", scale=2)
                show_chunks = gr.Checkbox(
                    label="Show raw retrieved chunks",
                    value=False,
                    scale=3,
                )

        # ── Right: sources ──────────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Retrieved Sources")
            sources_box = gr.Markdown(
                value="_Sources will appear here after your first question._",
                elem_id="sources-panel",
            )

    # ── Example questions ───────────────────────────────────────────────────────
    with gr.Row():
        gr.Markdown("#### Try an example:")

    gr.Examples(
        examples=[
            ["Who is the instructor and what are the office hours?"],
            ["What are the grading policies?"],
            ["What is the late submission policy?"],
            ["Explain what a language model is."],
            ["What is naive Bayes classification?"],
            ["How does tokenization work?"],
            ["What assignments are required for this course?"],
        ],
        inputs=msg_box,
        label="",
    )

    # ── Event wiring ────────────────────────────────────────────────────────────
    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot, show_chunks],
        outputs=[chatbot, sources_box, msg_box],
    )
    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, show_chunks],
        outputs=[chatbot, sources_box, msg_box],
    )
    clear_btn.click(
        fn=clear_all,
        outputs=[chatbot, sources_box, msg_box],
    )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )
