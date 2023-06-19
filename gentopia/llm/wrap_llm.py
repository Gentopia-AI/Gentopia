from typing import Generator

from gentopia.llm.base_llm import BaseLLM
from gentopia.manager.base_llm_manager import BaseServerInfo
from gentopia.model.completion_model import ChatCompletion, BaseCompletion
from gentopia.model.param_model import BaseParamModel
import requests


class WrapLLM(BaseLLM):
    server: BaseServerInfo

    def get_model_name(self) -> str:
        return self.server.llm_name

    def get_model_param(self) -> BaseParamModel:
        return self.params

    def completion(self, prompt) -> BaseCompletion:
        url = f"http://{self.server.host}:{self.server.port}/completion"
        data = {"prompt": prompt}
        response = requests.post(url, params=data, timeout=3000)
        x = response.json()
        print(x)
        return BaseCompletion(**x)

    def chat_completion(self, message) -> ChatCompletion:
        pass

    def stream_chat_completion(self, prompt) -> BaseCompletion:
        url = f"http://{self.server.host}:{self.server.port}/stream_chat_completion"
        data = {"prompt": prompt}
        generated_text = ""
        try:
            with requests.post(url, params=data, stream=True, timeout=3000) as r:
                for word in r.iter_lines(chunk_size=1):
                    new_text = word.decode('utf-8')
                    generated_text += new_text
                    yield BaseCompletion(state="success",
                                         content=new_text,
                                         prompt_token=len(prompt),
                                         completion_token=len(generated_text))
        except Exception:
            return BaseCompletion(state="error",
                                  content="",
                                  prompt_token=len(prompt),
                                  completion_token=len(generated_text))
