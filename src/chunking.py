import os
from typing import List, Dict, Any

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LEN
)

from src.document_loader import clean_text


# =========================
# Basic chunking
# =========================
def chunk_text_by_window(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_len: int = MIN_CHUNK_LEN
) -> List[str]:

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    text_len = len(text)

    while start < text_len:

        end = min(
            start + chunk_size,
            text_len
        )

        # tránh cắt giữa từ
        if end < text_len:

            split_at = text.rfind(
                " ",
                start + int(chunk_size * 0.7),
                end
            )

            if split_at != -1:
                end = split_at

        chunk = text[start:end].strip()

        if len(chunk) >= min_len:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(
            0,
            end - overlap
        )

    return chunks


# =========================
# Chunk pages
# =========================
def chunk_extracted_pages(
    pages: List[Dict[str, Any]],
    file_name: str
) -> List[Dict[str, Any]]:

    all_chunks = []

    chunk_id = 0

    source_file = os.path.basename(file_name)

    for page_item in pages:

        page = page_item.get("page")

        text = page_item.get("text", "")

        page_chunks = chunk_text_by_window(
            text=text
        )

        for chunk in page_chunks:

            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "page": page,
                    "source_file": source_file
                }
            )

            chunk_id += 1

    return all_chunks