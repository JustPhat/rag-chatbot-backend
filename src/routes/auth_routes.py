from fastapi import APIRouter, HTTPException

from src.schemas.auth_schema import (
    RegisterRequest,
    LoginRequest
)

from src.services.auth_service import (
    create_user,
    authenticate_user,
    create_access_token
)


router = APIRouter(
    prefix="/auth",
    tags=["Auth"]
)


@router.post("/register")
def register(
    request: RegisterRequest
):
    try:
        user = create_user(
            email=request.email,
            password=request.password,
            full_name=request.full_name
        )

        access_token = create_access_token(
            user_id=user["_id"],
            email=user["email"]
        )

        return {
            "message": "Register successful",
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user["_id"],
            "email": user["email"],
            "full_name": user.get("full_name")
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/login")
def login(
    request: LoginRequest
):
    user = authenticate_user(
        email=request.email,
        password=request.password
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    access_token = create_access_token(
        user_id=user["_id"],
        email=user["email"]
    )

    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user["_id"],
        "email": user["email"],
        "full_name": user.get("full_name")
    }