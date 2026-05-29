from fastapi import APIRouter
from fastapi import HTTPException

from fastapi import Depends
from src.dependencies.auth_dependency import get_current_user
from src.schemas.conversation_schema import UpdateConversationRequest

from src.conversation_service import (
    get_user_conversations,
    get_conversation,
    get_conversation_messages,
    delete_conversation_by_id,
    update_conversation_title

)

from src.document_service import (
    get_documents_by_conversation,
    count_documents_by_conversation
)

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)


# =========================
# Helper: Convert Mongo datetime
# =========================
def serialize_datetime(value):
    if value is None:
        return None

    return value.isoformat()


# =========================
# GET /conversations/
# =========================
@router.get("/")
def list_conversations(
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["_id"]
    """
    Lấy danh sách conversation của user hiện tại.
    Tạm thời dùng DEFAULT_USER_ID = user_124.
    Sau này sẽ thay bằng user_id lấy từ JWT.
    """

    conversations = get_user_conversations(
        user_id
    )

    result = []

    for conv in conversations:
        documents_count = count_documents_by_conversation(
            conversation_id=conv["_id"],
            user_id=user_id
        )
        result.append({
            "conversation_id": conv["_id"],
            "user_id": conv["user_id"],
            "title": conv.get("title"),
            "file_name": conv.get("file_name"),
            "documents_count": documents_count,
            "created_at": serialize_datetime(
                conv.get("created_at")
            ),
            "updated_at": serialize_datetime(
                conv.get("updated_at")
            )
        })

    return {
        "user_id": user_id,
        "total": len(result),
        "conversations": result
    }


# =========================
# GET /conversations/{conversation_id}
# =========================
@router.get("/{conversation_id}")
def get_conversation_detail(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["_id"]
    """
    Lấy metadata của một conversation.
    """

    conv = get_conversation(
        conversation_id
    )

    if not conv:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    if conv["user_id"] != user_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this conversation."
        )
    documents = get_documents_by_conversation(
        conversation_id=conversation_id,
        user_id=user_id
    )
    return {
        "conversation_id": conv["_id"],
        "user_id": conv["user_id"],
        "title": conv.get("title"),
        "file_name": conv.get("file_name"),
        "documents_count": len(documents),
        "documents": [
            {
                "document_id": doc["_id"],
                "file_name": doc.get("file_name"),
                "num_chunks": doc.get("num_chunks"),
                "model_name": doc.get("model_name"),
                "created_at": serialize_datetime(
                    doc.get("created_at")
                )
            }
            for doc in documents
        ],
        "created_at": serialize_datetime(
            conv.get("created_at")
        ),
        "updated_at": serialize_datetime(
            conv.get("updated_at")
        )
    }


# =========================
# GET /conversations/{conversation_id}/messages
# =========================
@router.get("/{conversation_id}/messages")
def get_messages_by_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["_id"]
    """
    Lấy toàn bộ lịch sử chat của một conversation.
    """

    conv = get_conversation(
        conversation_id
    )

    if not conv:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    if conv["user_id"] != user_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this conversation."
        )

    messages = get_conversation_messages(
        conversation_id
    )

    result = []

    for msg in messages:
        result.append({
            "message_id": str(msg["_id"]),
            "user_id": msg.get("user_id"),
            "conversation_id": msg.get("conversation_id"),
            "role": msg.get("role"),
            "content": msg.get("content"),
            "sources": msg.get("sources", []),
            "timestamp": serialize_datetime(
                msg.get("timestamp")
            )
        })

    return {
        "conversation_id": conversation_id,
        "file_name": conv.get("file_name"),
        "total": len(result),
        "messages": result
    }
# =========================
# PATCH /conversations/{conversation_id}
# =========================
@router.patch("/{conversation_id}")
def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Đổi tên một conversation của user hiện tại.
    """

    user_id = current_user["_id"]

    conv = get_conversation(
        conversation_id
    )

    if not conv:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    if conv["user_id"] != user_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to update this conversation."
        )

    new_title = request.title.strip()

    if not new_title:
        raise HTTPException(
            status_code=400,
            detail="Title cannot be empty."
        )

    updated = update_conversation_title(
        conversation_id=conversation_id,
        user_id=user_id,
        title=new_title
    )

    if not updated:
        raise HTTPException(
            status_code=400,
            detail="Conversation title was not updated."
        )

    return {
        "message": "Conversation updated successfully",
        "conversation_id": conversation_id,
        "title": new_title
    }
# =========================
# DELETE /conversations/{conversation_id}
# =========================
@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Xóa một conversation của user hiện tại.
    Đồng thời xóa toàn bộ messages thuộc conversation đó.
    Không xóa document_cache.
    """

    user_id = current_user["_id"]

    conv = get_conversation(
        conversation_id
    )

    if not conv:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    if conv["user_id"] != user_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to delete this conversation."
        )

    deleted = delete_conversation_by_id(
        conversation_id=conversation_id,
        user_id=user_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found or already deleted."
        )

    return {
        "message": "Conversation deleted successfully",
        "conversation_id": conversation_id
    }