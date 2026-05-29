from pymongo import MongoClient, ASCENDING, DESCENDING
from src.config import MONGO_URI, DB_NAME

# =========================
# MongoDB Client
# =========================
mongo_client = MongoClient(MONGO_URI)

# test connection
mongo_client.admin.command("ping")

db = mongo_client[DB_NAME]

# =========================
# Collections
# =========================
users_col = db["users"]

conversations_col = db["conversations"]

messages_col = db["messages"]

cache_col = db["document_cache"]

documents_col = db["documents"]
# =========================
# Indexes
# =========================

# conversations
conversations_col.create_index(
    [("user_id", ASCENDING), ("updated_at", DESCENDING)]
)

# messages
messages_col.create_index(
    [("conversation_id", ASCENDING), ("timestamp", ASCENDING)]
)

# cache
cache_col.create_index(
    [("cache_key", ASCENDING)],
    unique=True
)

cache_col.create_index(
    [
        ("user_id", ASCENDING),
        ("file_hash", ASCENDING),
        ("embedding_model", ASCENDING)
    ]
)
# documents
documents_col.create_index(
    [("conversation_id", ASCENDING)]
)

documents_col.create_index(
    [("user_id", ASCENDING), ("conversation_id", ASCENDING)]
)

documents_col.create_index(
    [("cache_key", ASCENDING)]
)
# users
users_col.create_index(
    [("email", ASCENDING)],
    unique=True
)
print("✅ MongoDB connected.")