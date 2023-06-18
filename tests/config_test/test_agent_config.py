import pytest

from gentopia.assembler.agent_assembler import AgentAssembler


class Test02:

    @pytest.mark.parametrize("file", ["data/agent.yaml"])
    def test_agent_tools(self, file):
        conf = AgentAssembler(file)
        agent = conf.get_agent()
        assert agent.tools.tools['Wikipedia'].description == "test"

    @pytest.mark.parametrize("file", ["data/agent.yaml"])
    def test_agent_prefix(self, file):
        conf = AgentAssembler(file)
        agent = conf.get_agent()
        assert agent.planner.prefix == "I want to"
