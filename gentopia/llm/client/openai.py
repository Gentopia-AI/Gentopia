import os
from typing import List

import openai

from gentopia.llm.base_llm import BaseLLM
from gentopia.llm.llm_info import *
from gentopia.model.completion_model import *
from gentopia.model.param_model import *


class OpenAIGPTClient(BaseLLM):
    def __init__(self, model_name: str, params: OpenAIParamModel, api_key: str = None):
        assert TYPES.get(model_name, None) == "OpenAI"
        self.api_key = api_key
        self.params = params
        self.model_name = model_name
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key is not None:
            openai.api_key = api_key

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_param(self) -> OpenAIParamModel:
        return self.params

    def completion(self, prompt: str, **kwargs) -> BaseCompletion:
        try:
            response = openai.ChatCompletion.create(
                n=self.params.n,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.params.temperature,
                max_tokens=self.params.max_tokens,
                top_p=self.params.top_p,
                frequency_penalty=self.params.frequency_penalty,
                presence_penalty=self.params.presence_penalty,
                **kwargs
            )
            return BaseCompletion(state="success",
                                  content=response.choices[0].message["content"],
                                  prompt_token=response.usage["prompt_tokens"],
                                  completion_token=response.usage["completion_tokens"])
        except Exception as exception:
            print("Exception:", exception)
            return BaseCompletion(state="error", content=exception)

    def chat_completion(self, message: List[dict]) -> ChatCompletion:
        try:
            response = openai.ChatCompletion.create(
                n=self.params.n,
                model=self.model_name,
                messages=message,
                temperature=self.params.temperature,
                max_tokens=self.params.max_tokens,
                top_p=self.params.top_p,
                frequency_penalty=self.params.frequency_penalty,
                presence_penalty=self.params.presence_penalty,
            )
            return ChatCompletion(state="success",
                                  role=response.choices[0].message["role"],
                                  content=response.choices[0].message["content"],
                                  prompt_token=response.usage["prompt_tokens"],
                                  completion_token=response.usage["completion_tokens"])
        except Exception as exception:
            print("Exception:", exception)
            return ChatCompletion(state="error", content=exception)

    def stream_chat_completion(self, prompt: str):
        # TODO: Implement stream_chat_completion
        raise NotImplementedError("TO BE IMPLEMENTED")
