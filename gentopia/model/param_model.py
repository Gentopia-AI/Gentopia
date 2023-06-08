from pydantic import BaseModel


class BaseParamModel(BaseModel):
    model_name: str


class OpenAIParamModel(BaseParamModel):
    """
    OpenAI API parameters
    """
    model_name: str = "gpt-4"
    max_tokens: int = 4032
    temperature: float = 0.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    n: int = 1
    stop: list = []
