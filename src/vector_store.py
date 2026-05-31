import os
import hashlib
import numpy as np
import faiss

from bson.binary import Binary

from src.embedding import (
    encode_texts,
    encode_query
)

from src.database import cache_col

from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD
)


# =========================
# File hash
# =========================
def compute_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


# =========================
# Cache key
# =========================
def get_cache_key(
    user_id,
    embedding_model,
    file_hash
):
    safe_model = (
        embedding_model
        .replace("/", "_")
        .replace(":", "_")
    )

    return (
        f"{user_id}_"
        f"{safe_model}_"
        f"{file_hash[:24]}"
    )


# =========================
# Serialize FAISS
# =========================
def serialize_faiss_index(index):
    index_array = faiss.serialize_index(
        index
    )

    return Binary(
        index_array.tobytes()
    )


# =========================
# Deserialize FAISS
# =========================
def deserialize_faiss_index(index_binary):
    index_array = np.frombuffer(
        index_binary,
        dtype=np.uint8
    )

    return faiss.deserialize_index(
        index_array
    )


# =========================
# Build FAISS
# =========================
def build_faiss_index(
    chunks,
    model_name: str = DEFAULT_EMBEDDING_MODEL
):
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = encode_texts(
        texts,
        model_name=model_name
    )

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dim
    )

    index.add(
        embeddings
    )

    return index


# =========================
# Save cache
# =========================
def save_document_cache(
    user_id,
    file_path,
    chunks,
    faiss_index,
    model_name: str = DEFAULT_EMBEDDING_MODEL
):
    file_hash = compute_file_hash(
        file_path
    )

    cache_key = get_cache_key(
        user_id,
        model_name,
        file_hash
    )

    doc = {
        "cache_key": cache_key,

        "user_id": user_id,

        # new standard field
        "model_name": model_name,

        # legacy-compatible field
        "embedding_model": model_name,

        "file_name": os.path.basename(
            file_path
        ),

        "file_hash": file_hash,

        "chunks": chunks,

        "faiss_index_bytes": serialize_faiss_index(
            faiss_index
        )
    }

    cache_col.update_one(
        {
            "cache_key": cache_key
        },
        {
            "$set": doc
        },
        upsert=True
    )

    return doc


# =========================
# Load cache
# =========================
def load_document_cache(
    user_id,
    file_path,
    model_name: str = DEFAULT_EMBEDDING_MODEL
):
    file_hash = compute_file_hash(
        file_path
    )

    cache_key = get_cache_key(
        user_id,
        model_name,
        file_hash
    )

    doc = cache_col.find_one(
        {
            "cache_key": cache_key
        }
    )

    if not doc:
        return None, None, None

    chunks = doc["chunks"]

    index = deserialize_faiss_index(
        doc["faiss_index_bytes"]
    )

    return doc, chunks, index


# =========================
# Load cache by cache_key
# =========================
def load_document_cache_by_key(
    cache_key
):
    doc = cache_col.find_one(
        {
            "cache_key": cache_key
        }
    )

    if not doc:
        return None, None, None

    chunks = doc["chunks"]

    index = deserialize_faiss_index(
        doc["faiss_index_bytes"]
    )

    return doc, chunks, index


# =========================
# Retrieval
# =========================
def retrieve_chunks(
    question,
    chunks,
    index,
    top_k=5,
    model_name: str = DEFAULT_EMBEDDING_MODEL
):
    if model_name is None:
        model_name = DEFAULT_EMBEDDING_MODEL
    q_emb = encode_query(
        question,
        model_name=model_name
    )

    k = min(
        top_k,
        len(chunks),
        index.ntotal
    )

    distances, indices = index.search(
        q_emb,
        k
    )

    retrieved = []

    for score, idx in zip(
        distances[0],
        indices[0]
    ):
        if idx == -1:
            continue

        if score < SIMILARITY_THRESHOLD:
            continue

        retrieved.append(
            {
                "score": float(score),
                "chunk": chunks[idx]
            }
        )

    return retrieved