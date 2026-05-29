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
    update_conversation_time
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
    current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user["_id"]

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
        # Load cache
        # =========================
        doc, chunks, index = load_document_cache(
            user_id,
            file_path
        )

        # =========================
        # Build new cache
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
                chunks
            )

            doc = save_document_cache(
                user_id,
                file_path,
                chunks,
                index
            )

        # =========================
        # Create new conversation
        # OR add file to existing conversation
        # =========================
        is_new_conversation = False

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

            conv_id = conversation_id

            update_conversation_time(
                conv_id
            )

        else:
            title = (
                f"Chat về "
                f"{doc['file_name']}"
            )

            conv_id = create_conversation(
                user_id=user_id,
                title=title,
                file_name=doc["file_name"],
                cache_key=doc["cache_key"],
                model_name=doc.get("model_name")
            )

            is_new_conversation = True

        # =========================
        # Create document metadata
        # =========================
        document = create_document_record(
            user_id=user_id,
            conversation_id=conv_id,
            file_name=doc["file_name"],
            cache_key=doc["cache_key"],
            num_chunks=len(chunks),
            model_name=doc.get("model_name")
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
            "is_new_conversation": is_new_conversation
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )