import uuid
from datetime import datetime, timedelta, timezone

from jose import jwt
from passlib.context import CryptContext
from pymongo.errors import DuplicateKeyError

from src.config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

from src.database import users_col


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def now_utc():
    return datetime.now(timezone.utc)


# =========================
# Password
# =========================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:
    return pwd_context.verify(
        plain_password,
        hashed_password
    )


# =========================
# JWT
# =========================
def create_access_token(
    user_id: str,
    email: str
) -> str:
    expire = now_utc() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire
    }

    token = jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )

    return token


# =========================
# User
# =========================
def create_user(
    email: str,
    password: str,
    full_name: str | None = None
):
    email = email.lower().strip()

    user_id = str(uuid.uuid4())

    user_doc = {
        "_id": user_id,
        "email": email,
        "password_hash": hash_password(password),
        "full_name": full_name,
        "created_at": now_utc(),
        "updated_at": now_utc()
    }

    try:
        users_col.insert_one(user_doc)

    except DuplicateKeyError:
        raise ValueError("Email already registered.")

    return user_doc


def authenticate_user(
    email: str,
    password: str
):
    email = email.lower().strip()

    user = users_col.find_one(
        {"email": email}
    )

    if not user:
        return None

    if not verify_password(
        password,
        user["password_hash"]
    ):
        return None

    return user