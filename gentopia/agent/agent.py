import re
from typing import AnyStr


class Agent:
    def __init__(self, name: AnyStr, tools, planner, solver):
        self.name = name
        self.tools = tools
        self.planner = planner
        self.solver = solver

        self.plans = []
        self.planner_evidences = {}
        self.worker_evidences = {}
        self.tool_counter = {}


    def run(self, input):
        # run is stateless, so we need to reset the evidences
        raise NotImplementedError

    # use planner evidences to assign tasks to respective workers.
    def _get_worker_evidences(self):
        for level in self.evidences_level:
            # TODO: Run simultaneously
            for e in level:
                tool_call = self.planner_evidences[e]
                if "[" not in tool_call:
                    self.worker_evidences[e] = tool_call
                    continue
                tool, tool_input = tool_call.split("[", 1)
                tool_input = tool_input[:-1]
                # find variables in input and replace with previous evidences
                for var in re.findall(r"#E\d+", tool_input):
                    if var in self.worker_evidences:
                        tool_input = tool_input.replace(var, "[" + self.worker_evidences[var] + "]")
                self.worker_evidences[e] = self.tools.run(tool, tool_input)
                self.tool_counter = self.tools.cost

    def _reinitialize(self):
        self.plans = []
        self.planner_evidences = {}
        self.worker_evidences = {}
        self.tool_counter = {}


if __name__ == '__main__':
    from gentopia.config import Config

    Config.load("config.yml")
