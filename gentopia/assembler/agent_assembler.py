from typing import Union, Dict, Optional
import os
import logging

import torch.cuda
from langchain import PromptTemplate
from gentopia.agent.base_agent import BaseAgent
from gentopia.assembler.config import Config
from gentopia.llm import HuggingfaceLLMClient, OpenAIGPTClient
from gentopia.llm.base_llm import BaseLLM
from gentopia.llm.llm_info import TYPES
from gentopia.manager.base_llm_manager import BaseLLMManager
from gentopia.model.agent_model import AgentType
from gentopia.model.param_model import OpenAIParamModel, HuggingfaceParamModel
from gentopia.tools import *
from gentopia.tools import BaseTool


# TODO: Install GentPool and load custom agents here.

class AgentAssembler:
    def __init__(self, file=None, config=None):
        if file is not None:
            self.config = Config.from_file(file)
        elif config is not None:
            self.config = Config.from_dict(config)

        self.plugins: Dict[str, Union[BaseAgent, BaseTool]] = dict()
        self.manager: Optional[BaseLLMManager] = None

    def get_agent(self, config=None):
        if config is None:
            config = self.config
        assert config is not None

        auth = config.get('auth', {})
        self._set_auth_env(auth)

        # Agent config
        name = config.get('name')
        _type = AgentType(config.get('type'))
        version = config.get('version', "")
        description = config.get('description', "")
        AgentClass = AgentType.get_agent_class(_type)
        prompt_template = self._get_prompt_template(config.get('prompt_template'))
        agent = AgentClass(
            name=name,
            type=_type,
            version=version,
            description=description,
            target_tasks=config.get('target_tasks', []),
            llm=self._get_llm(config['llm']),
            prompt_template=prompt_template,
            plugins=self._parse_plugins(config.get('plugins', []))
        )
        return agent

    def _get_llm(self, obj) -> Union[BaseLLM, Dict[str, BaseLLM]]:
        assert isinstance(obj, dict) or isinstance(obj, str)
        if isinstance(obj, dict) and ("model_name" not in obj):
            return {
                #list(item.keys())[0]: self._parse_llm(list(item.values())[0]) for item in obj
                key: self._parse_llm(obj[key]) for key in obj
            }
        else:
            return self._parse_llm(obj)

    def _parse_llm(self, obj) -> BaseLLM:
        if isinstance(obj, str):
            name = obj
            model_param = dict()
        else:
            name = obj['model_name']
            model_param = obj.get('params', dict())
        llm = None
        if TYPES.get(name, None) == "OpenAI":
            # key = obj.get('key', None)
            params = OpenAIParamModel(**model_param)
            llm = OpenAIGPTClient(model_name=name, params=params)
        elif TYPES.get(name, None) == "Huggingface":
            device = obj.get('device', 'gpu' if torch.cuda.is_available() else 'cpu')
            params = HuggingfaceParamModel(**model_param)
            llm = HuggingfaceLLMClient(model_name=name, params=params, device=device)
        if llm is None:
            raise ValueError(f"LLM {name} is not supported currently.")
        if self.manager is None:
            return llm
        return self.manager.get_llm(name, params, cls=HuggingfaceLLMClient, device=device)

    def _get_prompt_template(self, obj):
        assert isinstance(obj, dict) or isinstance(obj, PromptTemplate)
        if isinstance(obj, dict):
            return {
                key: self._parse_prompt_template(obj[key]) for key in obj
            }
        else:
            ans = self._parse_prompt_template(obj)
            return ans

    def _parse_prompt_template(self, obj):
        assert isinstance(obj, dict) or isinstance(obj, PromptTemplate)
        if isinstance(obj, PromptTemplate):
            return obj
        input_variables = obj['input_variables']
        template = obj['template']
        template_format = obj.get('template_format', 'f-string')
        validate_template = bool(obj.get('validate_template', True))
        return PromptTemplate(input_variables=input_variables, template=template, template_format=template_format,
                              validate_template=validate_template)

    def _parse_plugins(self, obj):
        assert isinstance(obj, list)
        result = []
        for plugin in obj:
            # If referring to a tool class then directly load it
            if issubclass(plugin.__class__, BaseTool):
                result.append(plugin)
                continue

            # Directly invoke already loaded plugin
            if plugin['name'] in self.plugins:
                _plugin = self.plugins[plugin['name']]
                result.append(_plugin)
                continue

            # Agent as plugin
            if plugin.get('type', "") in AgentType.__members__:
                agent = self.get_agent(plugin)
                result.append(agent)
                self.plugins[plugin['name']] = agent

            # Tool as plugin
            else:
                params = plugin.get('params', dict())
                tool = load_tools(plugin['name'])(**params)
                result.append(tool)
                self.plugins[plugin['name']] = tool
        return result

    def _set_auth_env(self, obj):
        for key in obj:
            os.environ[key] = obj.get(key)
