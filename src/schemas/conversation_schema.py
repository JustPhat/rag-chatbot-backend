from pydantic import BaseModel, Field


class UpdateConversationRequest(BaseModel):
    title: str = Field(
        min_length=1,
        max_length=120
    )