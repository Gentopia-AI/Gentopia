import signal
import time

import rich
from rich.box import Box

from gentopia.assembler.agent_assembler import AgentAssembler
from gentopia.assembler.config import Config
from gentopia.llm import HuggingfaceLLMClient
from gentopia.manager.local_llm_manager import LocalLLMManager
from transformers import GenerationConfig

from gentopia.model.param_model import HuggingfaceParamModel
from gentopia.output import enable_log
from gentopia.output.console_output import ConsoleOutput

import logging
# import warnings
#
# warnings.filterwarnings("ignore")

# 获取根logger对象
# root_logger = logging.getLogger()
#
# # 设置根logger级别为CRITICAL
# root_logger.setLevel(logging.CRITICAL)


def print_tree(obj, indent=0):
    for attr in dir(obj):
        if not attr.startswith('_'):
            value = getattr(obj, attr)
            if not callable(value):
                if not isinstance(value, dict) and not isinstance(value, list):
                    print('|   ' * indent + '|--', f'{attr}: {value}')
                else:
                    if not value:
                        print('|   ' * indent + '|--', f'{attr}: {value}')
                    print('|   ' * indent + '|--', f'{attr}:')
                if hasattr(value, '__dict__'):
                    print_tree(value, indent + 1)
                elif isinstance(value, list):
                    for item in value:
                        print_tree(item, indent + 1)
                elif isinstance(value, dict):
                    for key, item in value.items():
                        print_tree(item, indent + 1)


def ask(agent):
    out = ConsoleOutput()

    def handler(signum, frame):
        out.print("\n[red]Bye!")
        exit(0)

    signal.signal(signal.SIGINT, handler)
    while True:
        out.print("[green]User: ", end="")
        text = input()
        if text:
            response = agent.stream(text, output=out)
        else:
            response = agent.stream(output=out)

        out.done(_all=True)
        print("\n")


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)# calculate sqrt(10), and then calculate sqrt(100)
    # config = Config.load('main.yaml') # then tell me what is GIL in python
    # print(config)calculate sqrt(10),then tell me what is GIL in python, and then calculate sqrt(100)
    # exit(0)give me some sentences in markdown format
    enable_log(log_level='debug')

    assembler = AgentAssembler(file='configs/mathria.yaml')

    # # assembler.manager = LocalLLMManager()
    agent = assembler.get_agent()
    ask(agent)
    #
    # print(agent)
    # x = " What is Trump's current age raised to the 0.43 power?"
    # agent.run(x)
    # exit(0)

    # print(agent.llm.stream_chat_completion("1+1=?"))
    # print(agent.run("print hello world"))
    # print(agent.plugins)https://en.wikipedia.org/wiki/Camila_Morrone
# import time
from transformers.models.llama.modeling_llama import LlamaForCausalLM
# from gentopia.llm import HuggingfaceLLMClient
# from gentopia.llm.test_llm import TestLLM
# from gentopia.manager.local_llm_manager import LocalLLMManager
# from gentopia.model.param_model import BaseParamModel, HuggingfaceParamModel

# from gentopia.llm import HuggingfaceLLMClient
# from gentopia.manager.llm_client.local_llm_client import LocalLLMClient
# from gentopia.manager.local_llm_manager import LocalServerInfo
# from gentopia.model.param_model import HuggingfaceParamModel
#
# param = HuggingfaceParamModel(
#     model_name="wizardlm-13b",
#     temperature = 0.0,
#     top_p= 1.0,
#     max_new_tokens= 12,
#     # presence_penalty = 0.3
# )
#
# client = HuggingfaceLLMClient(model_name="wizardlm-13b", params=param, device="cpu")
# for i in client.stream_chat_completion("1+1=?"):
#     print(i)


# "User: hi\n"
# "Bot: hello\n"
# "User: how are you?\n"
# "Bot: I am fine\n"
#
# client = LocalLLMClient(server=LocalServerInfo(host="localhost", port=8000, llm_name="guanaco-7b", log_level="info"), llm=client)
# client.run()

# m = LocalLLMManager()
# param = HuggingfaceParamModel(
#     model_name="guanaco-7b"
# )
# llm = m.get_llm("guanaco-7b", param, HuggingfaceLLMClient, device="cpu")
# # input("Press Enter to continue...")
# print(llm.completion("1+1=?"))
# print("end")
