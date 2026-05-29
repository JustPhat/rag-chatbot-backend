import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# MongoDB
# =========================
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "ChatBot")

# =========================
# API Keys
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# Models
# =========================
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "AITeamVN/Vietnamese_Embedding"
)

LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "llama-3.1-8b-instant"
)

# =========================
# User
# =========================
DEFAULT_USER_ID = os.getenv(
    "DEFAULT_USER_ID",
    "user_124"
)

# =========================
# Chunking
# =========================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", 80))

# =========================
# Retrieval
# =========================
SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", 0.35)
)

# =========================
# Auth / JWT
# =========================
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "rag_chatbot_secret_2026"
)

JWT_ALGORITHM = os.getenv(
    "JWT_ALGORITHM",
    "HS256"
)

ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440)
)