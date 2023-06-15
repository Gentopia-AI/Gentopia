from typing import List, Union, Dict

from langchain import PromptTemplate

from gentopia.agent.base_agent import BaseAgent
from gentopia.config.config import Config
from gentopia.llm.base_llm import BaseLLM
from gentopia.llm.tmp_llm import TmpLLM
from gentopia.model.agent_model import AgentType


class AgentConfig:
    def __init__(self, file=None, config=None):
        if file is not None:
            self.config = Config.from_file(file)
        elif config is not None:
            self.config = Config.from_dict(config)

        self.plugins: Dict[Union[BaseAgent, List]] = []

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
        description = obj['description']
        model_param = obj['model_param']
        return TmpLLM(model_name=name, model_description=description, model_param=model_param)

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
            if i['type'] != 'tool':
                agent = self.get_agent(i)
                result.append(agent)
                self.plugins[i['name']] = agent
            else:
                pass
                # TODO: load tools
        return result
