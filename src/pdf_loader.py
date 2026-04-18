"""Extract text per page from PDF files using pypdf."""

from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: Path) -> list[dict]:
    """Load a PDF and return a list of page dicts with text and page number.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of dicts with keys ``page_number`` (1-indexed) and ``text``.
    """
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page_number": i + 1, "text": text})
    return pages


def load_all_pdfs(data_dir: Path) -> list[tuple[Path, list[dict]]]:
    """Load all PDFs from a directory.

    Args:
        data_dir: Directory that contains ``.pdf`` files.

    Returns:
        List of ``(pdf_path, pages)`` tuples.

    Raises:
        FileNotFoundError: If ``data_dir`` contains no PDF files.
    """
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. "
            "Place your PDF documents in the data/ directory before building the index."
        )
    return [(p, load_pdf(p)) for p in pdf_files]
