from gentopia.agent.agent import Agent
from gentopia.agent.nodes.Planner import Planner
from gentopia.agent.nodes.Solver import Solver
from gentopia.agent.tools_manager import ToolsManager
from gentopia.config.config import Config


class AgentConfig:
    def __init__(self, file=None, config=None):
        if file is not None:
            self.config = Config.from_file(file)
        elif config is not None:
            self.config = Config.from_dict(config)

    def get_agent(self):
        assert self.config is not None
        name = self.config['name']
        tools = ToolsManager(self.config['tools'])
        planner_config = self.config['planner']
        solver_config = self.config['solver']
        planner = Planner(tools, planner_config['prefix'], planner_config['suffix'], self.config['fewshot'],
                          planner_config['model_name'], planner_config['stop']
                          )
        solver = Solver(solver_config['prefix'], solver_config['suffix'],
                        solver_config['model_name'], solver_config['stop']
                        )
        return Agent(name, tools, planner, solver)
