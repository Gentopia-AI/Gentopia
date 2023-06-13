import re
from datetime import time
from typing import AnyStr

from gentopia.agent.agent import Agent
from gentopia.agent.parser import parse_plans, parse_planner_evidences
from gentopia.util.util import get_token_unit_price


class ReWoo(Agent):
    def __init__(self, name: AnyStr, tools, planner, solver):
        super().__init__(name, tools, planner, solver)
        self.planner_token_unit_price = get_token_unit_price(planner.model_name)
        self.solver_token_unit_price = get_token_unit_price(solver.model_name)
        self.tool_token_unit_price = get_token_unit_price("text-davinci-003")
        self.google_unit_price = 0.01

    def run(self, input):
        # run is stateless, so we need to reset the evidences
        self._reinitialize()
        result = {}
        st = time.time()
        # Plan
        planner_response = self.planner.run(input, log=True)
        plan = planner_response["output"]
        planner_log = planner_response["input"] + planner_response["output"]
        self.plans = parse_plans(plan)
        self.planner_evidences, self.evidences_level = parse_planner_evidences(plan)

        # Work
        self._get_worker_evidences()
        worker_log = ""
        for i in range(len(self.plans)):
            e = f"#E{i + 1}"
            worker_log += f"{self.plans[i]}\nEvidence:\n{self.worker_evidences[e]}\n"

        # Solve
        solver_response = self.solver.run(input, worker_log, log=True)
        output = solver_response["output"]
        solver_log = solver_response["input"] + solver_response["output"]

        result["wall_time"] = time.time() - st
        result["input"] = input
        result["output"] = output
        result["planner_log"] = planner_log
        result["worker_log"] = worker_log
        result["solver_log"] = solver_log
        result["tool_usage"] = self.tool_counter
        result["steps"] = len(self.plans) + 1
        result["total_tokens"] = planner_response["prompt_tokens"] + planner_response["completion_tokens"] \
                                 + solver_response["prompt_tokens"] + solver_response["completion_tokens"] \
                                 + self.tool_counter.get("LLM_token", 0) \
                                 + self.tool_counter.get("Calculator_token", 0)
        result["token_cost"] = self.planner_token_unit_price * (
                planner_response["prompt_tokens"] + planner_response["completion_tokens"]) \
                               + self.solver_token_unit_price * (
                                       solver_response["prompt_tokens"] + solver_response["completion_tokens"]) \
                               + self.tool_token_unit_price * (
                                       self.tool_counter.get("LLM_token", 0) + self.tool_counter.get(
                                   "Calculator_token", 0))
        result["tool_cost"] = self.tool_counter.get("Google", 0) * self.google_unit_price
        result["total_cost"] = result["token_cost"] + result["tool_cost"]

        return result

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
