from pydantic import BaseModel


class ChatRequest(BaseModel):

    conversation_id: str

    question: str

    search_mode: str = "Balanced"


class ChatResponse(BaseModel):

    conversation_id: str

    question: str

    answer: str

    search_mode: str

    top_k: int