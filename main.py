from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes.auth_routes import router as auth_router
from src.routes.chat_routes import router as chat_router
from src.routes.upload_routes import router as upload_router
from src.routes.conversation_routes import router as conversation_router

app = FastAPI(
    title="RAG Chatbot API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(conversation_router)


@app.get("/")
def root():
    return {
        "message": "RAG Chatbot Backend Running"
    }