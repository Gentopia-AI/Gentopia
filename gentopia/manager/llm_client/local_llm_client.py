from typing import AnyStr
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi import FastAPI

from gentopia.manager.llm_client.base_llm_client import BaseLLMClient
from gentopia.manager.server_info import LocalServerInfo

from gentopia.model.completion_model import BaseCompletion


class LocalLLMClient(BaseLLMClient):
    def __init__(self, server: LocalServerInfo, llm):
        self.server_info = server
        self.llm = llm
        self.router.add_api_route("/shutdown", self.shutdown, methods=["GET"])
        self.router.add_api_route("/completion", self.completion, methods=["POST"])
        self.router.add_api_route("/stream_chat_completion", self.stream_chat_completion, methods=["POST"])
        self.router.add_api_route("/test", self.completion, methods=["GET"])
        self.app = FastAPI()
        self.app.include_router(self.router)
        self.config = uvicorn.Config(self.app, host=server.host, port=server.port, log_level=server.log_level)
        self.server = uvicorn.Server(self.config)

    def run(self):
        self.server.run()

    def shutdown(self):
        del self.llm
        self.server.should_exit = True

    def completion(self, prompt: str) -> BaseCompletion:
        try:
            x = self.llm.completion(prompt)
        except Exception as e:
            print(e)
            raise e
        return x

    def test(self) -> str:
        return "test"

    def chat_completion(self, message) -> AnyStr:
        return "chat_completion is not supported now"

    def _stream(self, prompt):
        for x in self.llm.stream_chat_completion(prompt):
            print(x.content)
            yield x.content + "\n"

    def stream_chat_completion(self, prompt):
        return StreamingResponse(self._stream(prompt))
