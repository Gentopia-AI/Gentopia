from pydantic import BaseModel


class BaseCompletion(BaseModel):
    state: str  # "success" or "error"
    content: str
    prompt_token: int = 0
    completion_token: int = 0


class ChatCompletion(BaseCompletion):
    role: str = "assistant"  # "system" or "user" or "assistant"
