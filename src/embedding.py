import numpy as np

from sentence_transformers import SentenceTransformer

from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    SUPPORTED_EMBEDDING_MODELS
)


# =========================
# Model cache
# =========================
_embedding_model_cache = {}


# =========================
# Validate model
# =========================
def validate_embedding_model(
    model_name: str
):
    if model_name not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(
            f"Unsupported embedding model: {model_name}"
        )


# =========================
# Get embedding model
# =========================
def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL
):
    """
    Load embedding model theo model_name.
    Model được cache trong RAM để tránh load lại nhiều lần.
    """

    validate_embedding_model(
        model_name
    )

    if model_name not in _embedding_model_cache:

        print(
            f"⏳ Loading embedding model: "
            f"{model_name}"
        )

        _embedding_model_cache[model_name] = SentenceTransformer(
            model_name
        )

        print(
            f"✅ Loaded embedding model: "
            f"{model_name}"
        )

    return _embedding_model_cache[model_name]


# =========================
# Encode chunks
# =========================
def encode_texts(
    texts,
    model_name: str = DEFAULT_EMBEDDING_MODEL
):

    embedding_model = get_embedding_model(
        model_name
    )

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.astype(
        np.float32
    )


# =========================
# Encode query
# =========================
def encode_query(
    query: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL
):

    embedding_model = get_embedding_model(
        model_name
    )

    embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    return embedding.astype(
        np.float32
    )