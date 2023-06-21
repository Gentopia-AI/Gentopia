import os
from typing import List, Callable

import openai

from gentopia.llm.base_llm import BaseLLM
from gentopia.llm.llm_info import *
from gentopia.model.agent_model import AgentOutput
from gentopia.model.completion_model import *
from gentopia.model.param_model import *
import json


class OpenAIGPTClient(BaseLLM, BaseModel):
    model_name: str
    params: OpenAIParamModel

    def __init__(self, **data):
        super().__init__(**data)
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")

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

    def stream_chat_completion(self, message: List[dict]):
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
                stream=True
            )
            role = next(response).choices[0].delta["role"]
            messages = []
            ## TODO: Calculate prompt_token and completion_token
            for resp in response:
                messages.append(resp.choices[0].delta["content"])
                yield ChatCompletion(state="success",
                                     role=role,
                                     content=messages[-1],
                                     prompt_token=0,
                                     completion_token=0)
            return ChatCompletion(state="success",
                                  role=role,
                                  content=''.join(messages),
                                  prompt_token=0,
                                  completion_token=0)
        except Exception as exception:
            print("Exception:", exception)
            return ChatCompletion(state="error", content=exception)

    # TODO: Stream version, send to server once 'message' is updated
    def function_chat_completion(self, message: List[dict],
                                 function_map: Dict[str, Callable],
                                 function_schema: List[Dict]) -> ChatCompletionWithHistory:
        assert len(function_schema) == len(function_map)
        try:
            response = openai.ChatCompletion.create(
                n=self.params.n,
                model=self.model_name,
                messages=message,
                functions=function_schema,
                temperature=self.params.temperature,
                max_tokens=self.params.max_tokens,
                top_p=self.params.top_p,
                frequency_penalty=self.params.frequency_penalty,
                presence_penalty=self.params.presence_penalty,
            )
            response_message = response.choices[0]["message"]
            print("!!!!!", response_message)

            if response_message.get("function_call"):
                function_name = response_message["function_call"]["name"]
                fuction_to_call = function_map[function_name]
                function_args = json.loads(response_message["function_call"]["arguments"])
                function_response = fuction_to_call(**function_args)

                # Postprocess function response
                if isinstance(function_response, AgentOutput):
                    function_response = function_response.output

                message.append(dict(response_message))
                message.append({"role": "function",
                                "name": function_name,
                                "content": function_response})
                second_response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=message,
                )
                message.append(dict(second_response.choices[0].message))
                return ChatCompletionWithHistory(state="success",
                                                 role=second_response.choices[0].message["role"],
                                                 content=second_response.choices[0].message["content"],
                                                 message_scratchpad=message)
            else:
                message.append(dict(response_message))
                return ChatCompletionWithHistory(state="success",
                                                 role=response.choices[0].message["role"],
                                                 content=response.choices[0].message["content"],
                                                 message_scratchpad=message)

        except Exception as exception:
            print("Exception:", exception)
            return ChatCompletion(state="error", content=str(exception))
