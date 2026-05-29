import uuid

from datetime import datetime, timezone
from src.document_service import delete_documents_by_conversation
from src.database import (
    conversations_col,
    messages_col
)


# =========================
# Time
# =========================
def now_utc():

    return datetime.now(
        timezone.utc
    )


# =========================
# Create conversation
# =========================
def create_conversation(
    user_id: str,
    title: str,
    file_name: str | None = None,
    cache_key: str | None = None,
    model_name: str | None = None
):
    conv_id = str(uuid.uuid4())

    conv_doc = {
        "_id": conv_id,
        "user_id": user_id,
        "title": title,
        "file_name": file_name,      # legacy field, tạm giữ
        "cache_key": cache_key,      # legacy field, tạm giữ
        "model_name": model_name,    # chuẩn bị cho multi-model sau
        "created_at": now_utc(),
        "updated_at": now_utc()
    }

    conversations_col.insert_one(conv_doc)

    return conv_id


# =========================
# Get conversations
# =========================
def get_user_conversations(
    user_id
):

    cursor = conversations_col.find(
        {"user_id": user_id}
    ).sort("updated_at", -1)

    return list(cursor)


# =========================
# Get conversation
# =========================
def get_conversation(
    conv_id
):

    return conversations_col.find_one(
        {"_id": conv_id}
    )


# =========================
# Update time
# =========================
def update_conversation_time(
    conv_id
):

    conversations_col.update_one(
        {"_id": conv_id},
        {
            "$set": {
                "updated_at": now_utc()
            }
        }
    )


# =========================
# Save message
# =========================
def save_message(
    user_id,
    conv_id,
    role,
    content,
    sources=None
):

    doc = {

        "user_id": user_id,

        "conversation_id": conv_id,

        "role": role,

        "content": content,

        "sources": sources or [],

        "timestamp": now_utc()
    }

    messages_col.insert_one(doc)

    update_conversation_time(
        conv_id
    )


# =========================
# Get messages
# =========================
def get_conversation_messages(
    conv_id
):

    cursor = messages_col.find(
        {"conversation_id": conv_id}
    ).sort("timestamp", 1)

    return list(cursor)


# =========================
# Build Gradio history
# =========================
def build_chat_history(messages):
    history = []

    for msg in messages:
        if msg["role"] == "user":
            history.append({
                "role": "user",
                "content": msg["content"]
            })

        elif msg["role"] == "assistant":
            assistant_text = msg["content"]

            if msg.get("sources"):
                assistant_text += "\n\n---\n📚 Sources:\n"

                for source in msg["sources"]:
                    page_text = ""

                    if source.get("page"):
                        page_text = f"(Page {source['page']})"

                    assistant_text += (
                        f"\n• {page_text}\n"
                        f"{source['text'][:300]}...\n"
                    )

            history.append({
                "role": "assistant",
                "content": assistant_text
            })

    return history
# =========================
# Delete conversation
# =========================
def delete_conversation_by_id(
    conversation_id: str,
    user_id: str
) -> bool:
    """
    Xóa conversation và toàn bộ messages liên quan.
    Không xóa document_cache vì cache có thể được dùng bởi nhiều conversation.
    """

    conv = conversations_col.find_one(
        {
            "_id": conversation_id,
            "user_id": user_id
        }
    )

    if not conv:
        return False

    conversations_col.delete_one(
        {
            "_id": conversation_id,
            "user_id": user_id
        }
    )

    messages_col.delete_many(
        {
            "conversation_id": conversation_id,
            "user_id": user_id
        }
    )

    delete_documents_by_conversation(
        conversation_id=conversation_id,
        user_id=user_id
    )

    return True
# =========================
# Update conversation title
# =========================
def update_conversation_title(
    conversation_id: str,
    user_id: str,
    title: str
) -> bool:
    """
    Đổi tên conversation của user hiện tại.
    """

    result = conversations_col.update_one(
        {
            "_id": conversation_id,
            "user_id": user_id
        },
        {
            "$set": {
                "title": title,
                "updated_at": now_utc()
            }
        }
    )

    return result.modified_count > 0