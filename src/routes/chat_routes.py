from fastapi import APIRouter
from fastapi import HTTPException


from src.schemas.chat_schema import ChatRequest

from src.conversation_service import (
    get_conversation
)

from src.vector_store import (
    load_document_cache_by_key
)

from src.rag_service import (
    answer_question_multi_documents
)

from src.document_service import (
    get_documents_by_conversation
)

from fastapi import Depends
from src.dependencies.auth_dependency import get_current_user

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


# =========================
# Search mode mapping
# =========================
def get_top_k_from_mode(search_mode: str) -> int:

    mode = search_mode.lower().strip()

    if mode == "instant":
        return 3

    if mode == "balanced":
        return 5

    if mode == "deep":
        return 12

    return 5


# =========================
# Chat endpoint
# =========================
@router.post("/")
def chat_with_document(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):

    try:
        user_id = current_user["_id"]
        # =========================
        # Validate question
        # =========================
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        # =========================
        # Load conversation
        # =========================
        conv = get_conversation(
            request.conversation_id
        )

        if not conv:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found."
            )

        # Tạm thời vẫn dùng user cố định
        if conv["user_id"] != user_id:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this conversation."
            )

        # =========================
        # Load FAISS cache
        # =========================
        # =========================
        # Load all documents in conversation
        # =========================
        documents = get_documents_by_conversation(
            conversation_id=request.conversation_id,
            user_id=user_id
        )

        document_contexts = []

        # =========================
        # New multi-document path
        # =========================
        for document in documents:
            cache_key = document.get("cache_key")

            if not cache_key:
                continue

            cache_doc, chunks, index = load_document_cache_by_key(
                cache_key
            )

            if index is None or chunks is None:
                continue

            document_contexts.append({
                "file_name": document.get("file_name"),
                "chunks": chunks,
                "index": index
            })

        # =========================
        # Legacy fallback
        # For old conversations without documents collection
        # =========================
        if len(document_contexts) == 0:
            legacy_cache_key = conv.get("cache_key")

            if legacy_cache_key:
                cache_doc, chunks, index = load_document_cache_by_key(
                    legacy_cache_key
                )

                if index is not None and chunks is not None:
                    document_contexts.append({
                        "file_name": conv.get("file_name"),
                        "chunks": chunks,
                        "index": index
                    })

        if len(document_contexts) == 0:
            raise HTTPException(
                status_code=404,
                detail="No document cache found for this conversation."
            )
        # =========================
        # Search mode
        # =========================
        top_k = get_top_k_from_mode(
            request.search_mode
        )

        # =========================
        # RAG answer
        # =========================
        answer, history, sources = answer_question_multi_documents(
            question=question,
            document_contexts=document_contexts,
            top_k=top_k,
            conv_id=request.conversation_id,
            user_id=user_id
        )

        return {
            "conversation_id": request.conversation_id,
            "file_names": [
                item["file_name"]
                for item in document_contexts
            ],
            "documents_count": len(document_contexts),
            "question": question,
            "answer": answer,
            "sources": sources,
            "search_mode": request.search_mode,
            "top_k": top_k,
            "user_id": user_id
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )