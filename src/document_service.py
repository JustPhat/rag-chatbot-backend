import uuid
from datetime import datetime, timezone

from src.database import documents_col


def now_utc():
    return datetime.now(timezone.utc)


# =========================
# Create document record
# =========================
def create_document_record(
    user_id: str,
    conversation_id: str,
    file_name: str,
    cache_key: str,
    num_chunks: int,
    model_name: str | None = None
):
    """
    Lưu metadata của một file thuộc một conversation.

    Lưu ý:
    - FAISS index và chunks vẫn nằm trong document_cache.
    - documents chỉ lưu metadata để biết conversation có những file nào.
    """

    doc_id = str(uuid.uuid4())

    document_doc = {
        "_id": doc_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "file_name": file_name,
        "cache_key": cache_key,
        "num_chunks": num_chunks,
        "model_name": model_name,
        "created_at": now_utc(),
        "updated_at": now_utc()
    }

    documents_col.insert_one(document_doc)

    return document_doc


# =========================
# Get documents by conversation
# =========================
def get_documents_by_conversation(
    conversation_id: str,
    user_id: str
):
    cursor = documents_col.find(
        {
            "conversation_id": conversation_id,
            "user_id": user_id
        }
    ).sort("created_at", 1)

    return list(cursor)


# =========================
# Count documents by conversation
# =========================
def count_documents_by_conversation(
    conversation_id: str,
    user_id: str
) -> int:
    return documents_col.count_documents(
        {
            "conversation_id": conversation_id,
            "user_id": user_id
        }
    )


# =========================
# Delete documents by conversation
# =========================
def delete_documents_by_conversation(
    conversation_id: str,
    user_id: str
):
    """
    Xóa metadata documents khi xóa conversation.
    Không xóa document_cache ở bước này.
    """

    return documents_col.delete_many(
        {
            "conversation_id": conversation_id,
            "user_id": user_id
        }
    )