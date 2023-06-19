import multiprocessing
import signal
import socket
from typing import AnyStr, Tuple, List
from time import sleep
from gentopia.llm import HuggingfaceLLMClient
from gentopia.llm.base_llm import BaseLLM
from gentopia.llm.wrap_llm import WrapLLM
from gentopia.manager.base_llm_manager import BaseLLMManager, BaseServerInfo
from gentopia.manager.llm_client.local_llm_client import LocalLLMClient
from gentopia.manager.server_info import LocalServerInfo
from gentopia.model.param_model import BaseParamModel


def run_app(cls, llm_name, params, server, kwargs):
    llm = cls(model_name=llm_name, params=params, **kwargs)
    client = LocalLLMClient(server, llm)

    def exit_gracefully(signum, frame):
        print(f"Received signal {signum}. Exiting gracefully...")
        client.shutdown()

    signal.signal(signal.SIGINT, exit_gracefully)
    client.run()


class LocalLLMManager(BaseLLMManager):
    server: List[LocalServerInfo] = []
    _type = "LocalLLMManager"

    def _get_server(self, llm_name: AnyStr, model_params: BaseParamModel, **kwargs) -> Tuple[LocalServerInfo, bool]:
        server = self.find(llm_name, model_params, **kwargs)
        if server is not None:
            return server, False
        config = LocalServerInfo(host="localhost", port=self._get_free_port(), llm_name=llm_name,
                                 model_param=model_params, kwargs=kwargs)
        self.server.append(config)
        return config, True

    def wait(self, server: LocalServerInfo, limit: int = 10) -> bool:
        for i in range(limit):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((server.host, server.port))
                    return True
            except ConnectionRefusedError:
                print(f"Waiting for server {server.llm_name} to start...")
                sleep(0.3)
        return False

    def get_llm(self, llm_name: str, params: BaseParamModel, cls=None, **kwargs) -> WrapLLM:
        server, create = self._get_server(llm_name, params, **kwargs)
        if not create:
            print("Created!")
            return WrapLLM(model_name=llm_name, params=params, server=server)

        process = multiprocessing.Process(target=run_app, args=(cls, llm_name, params, server, kwargs))
        process.start()
        if not self.wait(server, 25):
            raise TimeoutError(f"Server {server.llm_name} did not start in time.")

        return WrapLLM(model_name=llm_name, params=params, server=server)

    def wrap_llm(self, llm: BaseLLM, **kwargs) -> WrapLLM:
        server, create = self._get_server(llm.model_name, llm.params, **kwargs)
        if not create:
            print("Created!")
            return WrapLLM(model_name=llm.model_name, params=llm.params, server=server)
        client = LocalLLMClient(server, llm)
        process = multiprocessing.Process(target=run_app, args=(client,))
        process.start()
        if not self.wait(server, 25):
            raise TimeoutError(f"Server {server.llm_name} did not start in time.")

        return WrapLLM(model_name=llm.model_name, params=llm.params, server=server)

    def _get_free_port(self):
        """
        Returns a free port number that can be used to bind a socket.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            return s.getsockname()[1]
