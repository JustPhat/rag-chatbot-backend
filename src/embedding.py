import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME

# =========================
# Load embedding model
# =========================
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print(
    f"✅ Loaded embedding model: "
    f"{EMBEDDING_MODEL_NAME}"
)


# =========================
# Encode chunks
# =========================
def encode_texts(texts):

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.astype(np.float32)


# =========================
# Encode query
# =========================
def encode_query(query: str):

    embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    return embedding.astype(np.float32)