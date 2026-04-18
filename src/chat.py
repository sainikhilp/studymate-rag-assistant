"""Interactive CLI REPL for the RAG chatbot."""

from src.rag import load_index, answer

_HELP = """Commands:
  exit / quit  — leave the chatbot
  clear        — reset conversation history
  sources      — show sources from the last answer"""


def _print_sources(sources: list[dict]) -> None:
    """Pretty-print a compact sources block."""
    if not sources:
        print("  (no sources)")
        return
    print("\nSources:")
    for i, s in enumerate(sources, 1):
        print(
            f"  [{i}] {s['source_file']}  p.{s['page_number']}  "
            f"score={s['score']:.4f}"
        )


def main() -> None:
    """Run the interactive RAG chatbot REPL."""
    print("Loading index …")
    index, metadata = load_index()
    print("Index loaded. Type your question, or 'help' for commands.\n")

    history: list[dict] = []
    last_sources: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lowered = user_input.lower()

        if lowered in ("exit", "quit"):
            print("Goodbye!")
            break

        if lowered == "clear":
            history.clear()
            last_sources = []
            print("Conversation history cleared.\n")
            continue

        if lowered == "sources":
            _print_sources(last_sources)
            print()
            continue

        if lowered == "help":
            print(_HELP + "\n")
            continue

        result = answer(user_input, index, metadata, history=history)
        reply = result["answer"]
        last_sources = result["sources"]

        print(f"\nAssistant: {reply}\n")
        _print_sources(last_sources)
        print()

        # Append to history for multi-turn context (use plain question, not context-stuffed message)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
