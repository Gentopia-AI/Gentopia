from typing import Union, Dict

from langchain import PromptTemplate

from gentopia.agent.base_agent import BaseAgent
from gentopia.config.config import Config
from gentopia.llm import HuggingfaceLLMClient, OpenAIGPTClient
from gentopia.llm.base_llm import BaseLLM
from gentopia.llm.llm_info import TYPES
from gentopia.model.agent_model import AgentType
from gentopia.model.param_model import OpenAIParamModel, HuggingfaceParamModel
from gentopia.tools import *
from gentopia.tools import BaseTool


class AgentConfig:
    def __init__(self, file=None, config=None):
        if file is not None:
            self.config = Config.from_file(file)
        elif config is not None:
            self.config = Config.from_dict(config)

        self.plugins: Dict[str, Union[BaseAgent, BaseTool]] = dict()

    def get_agent(self, config=None):
        if config is None:
            config = self.config
        assert config is not None
        name = config['name']
        _type = AgentType(config['type'])
        version = config['version']
        description = config['description']
        AgentClass = AgentType.get_agent_class(_type)
        prompt_template = self.get_prompt_template(config['prompt_template'])
        agent = AgentClass(
            name=name,
            type=_type,
            version=version,
            description=description,
            target_tasks=config.get('target_tasks', []),
            llm=self.get_llm(config['llm']),
            prompt_template=prompt_template,
            plugins=self.parse_plugins(config.get('plugins', []))
        )
        return agent

    def get_llm(self, obj) -> Union[BaseLLM, Dict[str, BaseLLM]]:
        assert isinstance(obj, dict) or isinstance(obj, list)
        if isinstance(obj, list):
            return {
                list(item.keys())[0]: self.parse_llm(list(item.values())[0]) for item in obj
            }
        else:
            return self.parse_llm(obj)

    def parse_llm(self, obj) -> BaseLLM:
        name = obj['name']
        model_param = obj.get('model_param', dict())
        if TYPES.get(name, None) == "OpenAI":
            key = obj.get('key', None)
            params = OpenAIParamModel(**model_param)
            return OpenAIGPTClient(model_name=name, params=params, api_key=key)
        device = obj.get('device', 'cpu')
        params = HuggingfaceParamModel(**model_param)
        return HuggingfaceLLMClient(model_name=name, model_param=params, device=device)

    def get_prompt_template(self, obj):
        assert isinstance(obj, dict) or isinstance(obj, list)
        if isinstance(obj, list):
            return {
                list(item.keys())[0]: self.parse_prompt_template(list(item.values())[0]) for item in obj
            }
        else:
            ans = self.parse_prompt_template(obj)
            return ans

    def parse_prompt_template(self, obj):
        assert isinstance(obj, dict)
        input_variables = obj['input_variables']
        template = obj['template']
        template_format = obj.get('template_format', 'f-string')
        validate_template = bool(obj.get('validate_template', True))
        return PromptTemplate(input_variables=input_variables, template=template, template_format=template_format,
                              validate_template=validate_template)


    def parse_plugins(self, obj):
        assert isinstance(obj, list)
        result = []
        for i in obj:
            if i['name'] in self.plugins:
                _plugin = self.plugins[i['name']]
                result.append(_plugin)
                continue
            if 'llm' in i:
                agent = self.get_agent(i)
                result.append(agent)
                self.plugins[i['name']] = agent
            else:
                params = i.get('tool_param', dict())
                tool = load_tools(i['type'])(**params)
                result.append(tool)
                self.plugins[i['name']] = tool
        return result
