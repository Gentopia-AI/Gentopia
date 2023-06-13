from gentopia.llm.base_llm import BaseLLM
from gentopia.model.completion_model import *
from gentopia.model.param_model import *
from gentopia.llm.llm_info import *
from typing import List
import openai
import os


class OpenAIGPTClient(BaseLLM):
    def __init__(self, params: OpenAIParamModel, api_key: str = None):
        assert TYPES.get(params.model_name, None) == "OpenAI"
        self.api_key = api_key
        self.params = params
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key is not None:
            openai.api_key = api_key

    def get_model_name(self):
        return self.params.model

    def get_model_param(self):
        return self.params

    def completion(self, prompt: str):
        try:
            response = openai.ChatCompletion.create(
                n=self.params.n,
                model=self.params.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.params.temperature,
                max_tokens=self.params.max_tokens,
                top_p=self.params.top_p,
                frequency_penalty=self.params.frequency_penalty,
                presence_penalty=self.params.presence_penalty,
            )
            return BaseCompletion(state="success",
                                  content=response.choices[0].message["content"],
                                  prompt_token=response.usage["prompt_tokens"],
                                  completion_token=response.usage["completion_tokens"])
        except Exception as exception:
            print("Exception:", exception)
            return BaseCompletion(state="error", content=exception)

    def chat_completion(self, message: List[dict]):
        try:
            response = openai.ChatCompletion.create(
                n=self.params.n,
                model=self.params.model_name,
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
        raise NotImplementedError("TO BE IMPLEMENTED")
