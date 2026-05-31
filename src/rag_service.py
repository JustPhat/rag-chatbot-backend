from groq import Groq

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL_NAME
)

from src.vector_store import (
    retrieve_chunks
)

from src.conversation_service import (
    save_message,
    get_conversation_messages,
    build_chat_history
)


groq_client = Groq(
    api_key=GROQ_API_KEY
)


# =========================
# Chunk helpers
# =========================
def get_chunk_text(chunk):
    """
    Hỗ trợ cả 2 dạng chunk:
    - dict: {"text": "...", "page": ...}
    - str: "..."
    """

    if isinstance(chunk, dict):
        return chunk.get("text", "")

    return str(chunk)


def get_chunk_page(chunk):
    if isinstance(chunk, dict):
        return chunk.get("page")

    return None


# =========================
# Build context - single document
# =========================
def build_context(
    retrieved_chunks
):

    context_parts = []

    for i, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        chunk = item["chunk"]

        page = get_chunk_page(
            chunk
        )

        page_text = ""

        if page:

            page_text = (
                f"(Trang {page})"
            )

        context_parts.append(
            f"[{i}] {page_text}\n"
            f"{get_chunk_text(chunk)}"
        )

    return "\n\n".join(
        context_parts
    )


# =========================
# Build context - multi documents
# =========================
def build_context_multi_documents(
    retrieved_sources
):

    context_parts = []

    for i, source in enumerate(
        retrieved_sources,
        start=1
    ):

        file_name = source.get(
            "file_name",
            "Unknown file"
        )

        page = source.get(
            "page"
        )

        page_text = ""

        if page:

            page_text = (
                f"Trang {page}"
            )
        else:

            page_text = (
                "Không có số trang"
            )

        context_parts.append(
            f"[{i}] File: {file_name} | {page_text}\n"
            f"{source.get('text', '')}"
        )

    return "\n\n".join(
        context_parts
    )


# =========================
# Ask RAG - single document
# =========================
def answer_question(
    question,
    chunks,
    index,
    top_k,
    conv_id,
    user_id,
    model_name=None
):

    retrieved = retrieve_chunks(
        question=question,
        chunks=chunks,
        index=index,
        top_k=top_k
    )

    if len(retrieved) == 0:

        no_answer = (
            "Tôi không tìm thấy "
            "thông tin phù hợp "
            "trong tài liệu."
        )

        save_message(
            user_id,
            conv_id,
            "user",
            question
        )

        save_message(
            user_id,
            conv_id,
            "assistant",
            no_answer
        )

        msgs = get_conversation_messages(
            conv_id
        )

        return (
            no_answer,
            build_chat_history(msgs),
            []
        )

    context = build_context(
        retrieved
    )

    deep_instruction = ""

    if top_k >= 10:

        deep_instruction = """
- Hãy tổng hợp đầy đủ tất cả thông tin liên quan.
- Có thể kết hợp thông tin từ nhiều đoạn khác nhau.
- Ưu tiên độ đầy đủ hơn tính ngắn gọn.
"""

    prompt = f"""
Bạn là trợ lý AI trả lời dựa trên tài liệu.

Chỉ dùng thông tin trong tài liệu.
{deep_instruction}
====================
TÀI LIỆU:
{context}
====================

CÂU HỎI:
{question}

TRẢ LỜI:
"""

    response = (
        groq_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=600
        )
    )

    answer = (
        response
        .choices[0]
        .message.content
        .strip()
    )

    sources = []

    for item in retrieved:

        chunk = item["chunk"]

        sources.append(
            {
                "text":
                    get_chunk_text(chunk),

                "page":
                    get_chunk_page(chunk),

                "score":
                    round(
                        item["score"],
                        4
                    )
            }
        )

    save_message(
        user_id,
        conv_id,
        "user",
        question
    )

    save_message(
        user_id,
        conv_id,
        "assistant",
        answer,
        sources=sources
    )

    msgs = get_conversation_messages(
        conv_id
    )

    history = build_chat_history(
        msgs
    )

    return answer, history, sources


# =========================
# Retrieve from one document
# =========================
def retrieve_from_one_document(
    question,
    chunks,
    index,
    top_k,
    file_name,
    model_name=None
):
    """
    Retrieve top_k chunks từ một document.
    Sau đó gắn thêm file_name vào mỗi source.
    """

    retrieved = retrieve_chunks(
        question=question,
        chunks=chunks,
        index=index,
        top_k=top_k,
        model_name=model_name
    )

    sources = []

    for item in retrieved:

        chunk = item["chunk"]

        text = get_chunk_text(
            chunk
        )

        if not text:
            continue

        sources.append(
            {
                "text":
                    text,

                "page":
                    get_chunk_page(chunk),

                "score":
                    round(
                        item["score"],
                        4
                    ),

                "file_name":
                    file_name
            }
        )

    return sources


# =========================
# Ask RAG - multi documents
# =========================
def answer_question_multi_documents(
    question,
    document_contexts,
    top_k,
    conv_id,
    user_id
):
    """
    document_contexts format:

    [
        {
            "file_name": "hanoi_clean.txt",
            "chunks": [...],
            "index": faiss_index
        },
        {
            "file_name": "Hồ_Chí_Minh.pdf",
            "chunks": [...],
            "index": faiss_index
        }
    ]
    """

    all_sources = []

    # =========================
    # Retrieve from all documents
    # =========================
    for doc_ctx in document_contexts:

        file_name = doc_ctx.get(
            "file_name",
            "Unknown file"
        )

        chunks = doc_ctx.get(
            "chunks"
        )

        index = doc_ctx.get(
            "index"
        )

        model_name = doc_ctx.get("model_name")

        if not chunks or index is None:
            continue

        document_sources = retrieve_from_one_document(
            question=question,
            chunks=chunks,
            index=index,
            top_k=top_k,
            file_name=file_name,
            model_name=model_name
        )

        all_sources.extend(
            document_sources
        )

    # =========================
    # Sort by score and keep final top_k
    # =========================
    all_sources = sorted(
        all_sources,
        key=lambda item: item.get("score", 0),
        reverse=True
    )

    sources = all_sources[:top_k]

    # =========================
    # No retrieved context
    # =========================
    if len(sources) == 0:

        no_answer = (
            "Tôi không tìm thấy "
            "thông tin phù hợp "
            "trong các tài liệu đã được cung cấp."
        )

        save_message(
            user_id,
            conv_id,
            "user",
            question
        )

        save_message(
            user_id,
            conv_id,
            "assistant",
            no_answer,
            sources=[]
        )

        msgs = get_conversation_messages(
            conv_id
        )

        return (
            no_answer,
            build_chat_history(msgs),
            []
        )

    # =========================
    # Build context
    # =========================
    context = build_context_multi_documents(
        sources
    )

    deep_instruction = ""

    if top_k >= 10:

        deep_instruction = """
- Hãy tổng hợp đầy đủ tất cả thông tin liên quan.
- Có thể kết hợp thông tin từ nhiều đoạn và nhiều file khác nhau.
- Ưu tiên độ đầy đủ hơn tính ngắn gọn.
"""

    prompt = f"""
Bạn là trợ lý AI trả lời dựa trên nhiều tài liệu do người dùng cung cấp.

Chỉ dùng thông tin trong phần TÀI LIỆU.
Nếu tài liệu không có thông tin, hãy nói rõ là không có thông tin trong tài liệu.
Không tự bịa thêm thông tin ngoài tài liệu.
{deep_instruction}

====================
TÀI LIỆU:
{context}
====================

CÂU HỎI:
{question}

TRẢ LỜI:
"""

    response = (
        groq_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=800
        )
    )

    answer = (
        response
        .choices[0]
        .message.content
        .strip()
    )

    # =========================
    # Save messages
    # =========================
    save_message(
        user_id,
        conv_id,
        "user",
        question
    )

    save_message(
        user_id,
        conv_id,
        "assistant",
        answer,
        sources=sources
    )

    msgs = get_conversation_messages(
        conv_id
    )

    history = build_chat_history(
        msgs
    )

    return answer, history, sources