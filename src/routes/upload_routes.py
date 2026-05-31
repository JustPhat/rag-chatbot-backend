from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException
from fastapi import Depends
from fastapi import Form

from src.dependencies.auth_dependency import get_current_user
from src.document_service import create_document_record

import shutil
import os

from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    SUPPORTED_EMBEDDING_MODELS
)

from src.document_loader import (
    extract_text_from_file
)

from src.chunking import (
    chunk_extracted_pages
)

from src.vector_store import (
    load_document_cache,
    build_faiss_index,
    save_document_cache
)

from src.conversation_service import (
    create_conversation,
    get_conversation,
    update_conversation_time,
    generate_unique_conversation_title
)


router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)


UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


@router.post("/")
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str | None = Form(None),
    model_name: str = Form(DEFAULT_EMBEDDING_MODEL),
    current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user["_id"]

        # =========================
        # Determine embedding model
        # =========================
        is_new_conversation = False
        conv = None

        if conversation_id:
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
                    detail="You do not have permission to add file to this conversation."
                )

            # Khi thêm file vào chat cũ:
            # luôn dùng model của conversation,
            # không dùng model_name frontend gửi lên.
            final_model_name = (
                conv.get("model_name")
                or DEFAULT_EMBEDDING_MODEL
            )

        else:
            # Khi tạo chat mới:
            # cho phép user chọn model.
            if model_name not in SUPPORTED_EMBEDDING_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported embedding model."
                )

            final_model_name = model_name
            is_new_conversation = True

        # =========================
        # Save uploaded file
        # =========================
        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        # =========================
        # Load cache by selected model
        # =========================
        doc, chunks, index = load_document_cache(
            user_id,
            file_path,
            model_name=final_model_name
        )

        # =========================
        # Build new cache if not exists
        # =========================
        if index is None:
            pages = extract_text_from_file(
                file_path
            )

            chunks = chunk_extracted_pages(
                pages,
                file.filename
            )

            if len(chunks) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot create chunks"
                )

            index = build_faiss_index(
                chunks,
                model_name=final_model_name
            )

            doc = save_document_cache(
                user_id,
                file_path,
                chunks,
                index,
                model_name=final_model_name
            )

        # =========================
        # Create new conversation
        # OR add file to existing conversation
        # =========================
        if conversation_id:
            conv_id = conversation_id

            update_conversation_time(
                conv_id
            )

        else:
            base_title = (
                f"Chat về "
                f"{doc['file_name']}"
            )

            title = generate_unique_conversation_title(
                user_id=user_id,
                base_title=base_title
            )

            conv_id = create_conversation(
                user_id=user_id,
                title=title,
                file_name=doc["file_name"],
                cache_key=doc["cache_key"],
                model_name=final_model_name
            )

        # =========================
        # Create document metadata
        # =========================
        document = create_document_record(
            user_id=user_id,
            conversation_id=conv_id,
            file_name=doc["file_name"],
            cache_key=doc["cache_key"],
            num_chunks=len(chunks),
            model_name=final_model_name
        )

        return {
            "message": (
                "Conversation created and file uploaded successfully"
                if is_new_conversation
                else "File added to existing conversation successfully"
            ),
            "conversation_id": conv_id,
            "document_id": document["_id"],
            "file_name": doc["file_name"],
            "num_chunks": len(chunks),
            "user_id": user_id,
            "is_new_conversation": is_new_conversation,
            "model_name": final_model_name,
            "model_label": SUPPORTED_EMBEDDING_MODELS.get(
                final_model_name,
                {}
            ).get(
                "label",
                final_model_name
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )