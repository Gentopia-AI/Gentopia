import os
from typing import Union, Dict, Optional

import torch.cuda

from gentopia.prompt import PromptTemplate
from gentopia.agent.base_agent import BaseAgent
from gentopia.assembler.config import Config
from gentopia.llm import HuggingfaceLLMClient, OpenAIGPTClient
from gentopia.llm.base_llm import BaseLLM
from gentopia.llm.llm_info import TYPES
from gentopia.manager.base_llm_manager import BaseLLMManager
from gentopia.memory.api import MemoryWrapper
from gentopia.memory.api import create_memory
from gentopia.model.agent_model import AgentType
from gentopia.model.param_model import OpenAIParamModel, HuggingfaceParamModel
from gentopia.tools import *
from gentopia.tools import BaseTool
from gentopia.tools.basetool import ToolMetaclass


class AgentAssembler:
    """
        This class is responsible for assembling an agent instance from a configuration file or dictionary and its dependencies.

        :param file: A path to a configuration file.
        :type file: str, optional
        :param config: A configuration dictionary.
        :type config: dict, optional
    """
    def __init__(self, file=None, config=None):
        """
            Constructor method.

            Initializes an instance of the AgentAssembler class.

            :param file: A path to a configuration file.
            :type file: str, optional
            :param config: A configuration dictionary.
            :type config: dict, optional
        """
        if file is not None:
            self.config = Config.from_file(file)
        elif config is not None:
            self.config = Config.from_dict(config)

        self.plugins: Dict[str, Union[BaseAgent, BaseTool]] = dict()
        self.manager: Optional[BaseLLMManager] = None

    def get_agent(self, config=None):
        """
            This method returns an agent instance based on the provided configuration.

            :param config: A configuration dictionary.
            :type config: dict, optional
            :raises AssertionError: If the configuration is None.
            :return: An agent instance.
            :rtype: BaseAgent
        """
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
            plugins=self._parse_plugins(config.get('plugins', [])),
            memory=self._parse_memory(config.get('memory', [])) # initialize memory
        )
        return agent

    def _parse_memory(self, obj) -> MemoryWrapper:
        """
            This method parses the memory configuration and returns a memory wrapper instance.

            :param obj: A configuration dictionary containing memory parameters.
            :type obj: dict
            :return: A memory wrapper instance.
            :rtype: MemoryWrapper
        """
        if obj == []:
            return None
        memory_type = obj["memory_type"] # memory_type: ["pinecone"]
        return create_memory(memory_type, obj['threshold_1'], obj['threshold_2'], **obj["params"]) # params of memory.
        # Different memories may have different params


    def _get_llm(self, obj) -> Union[BaseLLM, Dict[str, BaseLLM]]:
        """
            This method returns a language model manager (LLM) instance based on the provided configuration.

            :param obj: A configuration dictionary or string.
            :type obj: dict or str
            :raises AssertionError: If the configuration is not a dictionary or string.
            :raises ValueError: If the specified LLM is not supported.
            :return: An LLM instance or dictionary of LLM instances.
            :rtype: Union[BaseLLM, Dict[str, BaseLLM]]
        """
        assert isinstance(obj, dict) or isinstance(obj, str)
        if isinstance(obj, dict) and ("model_name" not in obj):
            return {
                # list(item.keys())[0]: self._parse_llm(list(item.values())[0]) for item in obj
                key: self._parse_llm(obj[key]) for key in obj
            }
        else:
            return self._parse_llm(obj)

    def _parse_llm(self, obj) -> BaseLLM:
        """
            This method parses the Language Model Manager (LLM) configuration and returns an LLM instance.

            :param obj: A configuration dictionary or string.
            :type obj: dict or str
            :raises ValueError: If the specified LLM is not supported.
            :return: An LLM instance.
            :rtype: BaseLLM
        """
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
        """
            This method returns a prompt template instance based on the provided configuration.

            :param obj: A configuration dictionary or prompt template instance.
            :type obj: dict or PromptTemplate
            :raises AssertionError: If the configuration is not a dictionary or prompt template instance.
            :return: A prompt template instance.
            :rtype: PromptTemplate
        """
        assert isinstance(obj, dict) or isinstance(obj, PromptTemplate)
        if isinstance(obj, dict):
            return {
                key: self._parse_prompt_template(obj[key]) for key in obj
            }
        else:
            ans = self._parse_prompt_template(obj)
            return ans

    def _parse_prompt_template(self, obj):
        """
            This method parses the prompt template configuration and returns a prompt template instance.

            :param obj: A configuration dictionary or prompt template instance.
            :type obj: dict or PromptTemplate
            :raises AssertionError: If the configuration is not a dictionary or prompt template instance.
            :return: A prompt template instance.
            :rtype: PromptTemplate
        """
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
        """
            This method parses the plugin configuration and returns a list of plugin instances.

            :param obj: A list of plugin configuration dictionaries.
            :type obj: list
            :raises AssertionError: If the configuration is not a list.
            :return: A list of plugin instances.
            :rtype: list
        """
        assert isinstance(obj, list)
        result = []
        for plugin in obj:
            # If referring to a tool class then directly load it
            if issubclass(plugin.__class__, ToolMetaclass):
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
        """
            This method sets environment variables for authentication.

            :param obj: A dictionary containing authentication information.
            :type obj: dict
        """
        for key in obj:
            os.environ[key] = obj.get(key)
