import pytest

from gentopia.config.agent_config import AgentConfig


class Test02:

    @pytest.mark.parametrize("file", ["data/agent.yaml"])
    def test_agent_tools(self, file):
        conf = AgentConfig(file)
        agent = conf.get_agent()
        assert agent.tools.tools['Wikipedia'].description == "test"

    @pytest.mark.parametrize("file", ["data/agent.yaml"])
    def test_agent_prefix(self, file):
        conf = AgentConfig(file)
        agent = conf.get_agent()
        assert agent.planner.prefix == "I want to"
