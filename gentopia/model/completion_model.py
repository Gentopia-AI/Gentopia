from pydantic import BaseModel
from typing import List, Dict


class BaseCompletion(BaseModel):
    state: str  # "success" or "error"
    content: str
    prompt_token: int = 0
    completion_token: int = 0

    def to_dict(self):
        return dict(
            state=self.state,
            content=self.content,
            prompt_token=self.prompt_token,
            completion_token=self.completion_token,
        )


class ChatCompletion(BaseCompletion):
    role: str = "assistant"  # "system" or "user" or "assistant"


class ChatCompletionWithHistory(ChatCompletion):
    """Used for function call API"""
    message_scratchpad: List[Dict] = []
    plugin_cost: float = 0.0
    plugin_token: float = 0.0
