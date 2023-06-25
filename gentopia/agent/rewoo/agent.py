import logging
import re
from typing import List, Dict, Union, Optional

from langchain import PromptTemplate

from gentopia.agent.base_agent import BaseAgent
from gentopia.agent.rewoo.nodes.Planner import Planner
from gentopia.agent.rewoo.nodes.Solver import Solver
from gentopia.llm.base_llm import BaseLLM
from gentopia.model.agent_model import AgentType
from gentopia.output.base_output import BaseOutput
from gentopia.tools import BaseTool
from gentopia.utils.cost_helpers import *
from gentopia.utils.text_helpers import *


class RewooAgent(BaseAgent):
    name: str = "RewooAgent"
    type: AgentType = AgentType.rewoo
    version: str = ""
    description: str
    target_tasks: list[str] = []
    llm: Union[BaseLLM, Dict[str, BaseLLM]]  # {"Planner": xxx, "Solver": xxx}
    prompt_template: Dict[str, PromptTemplate]  # {"Planner": xxx, "Solver": xxx}
    plugins: List[Union[BaseTool, BaseAgent]]
    examples: Dict[str, Union[str, List[str]]] = dict()
    # logger = logging.getLogger('application')

    def _get_llms(self):
        if isinstance(self.llm, BaseLLM):
            return {"Planner": self.llm, "Solver": self.llm}
        elif isinstance(self.llm, dict) and "Planner" in self.llm and "Solver" in self.llm:
            return {"Planner": self.llm["Planner"], "Solver": self.llm["Solver"]}
        else:
            raise ValueError("llm must be a BaseLLM or a dict with Planner and Solver.")

    def _parse_plan_map(self, planner_response: str) -> List[dict[str, List[str]]]:
        """
        Parse planner output. It should be an n-to-n mapping from Plans to *Es.
        This is because sometimes LLM cannot follow the strict output format.
        Example:
            *Plan1
            *E1
            *E2
        should result in: {"Plan1": ["*E1", "*E2"]}
        Or:
            *Plan1
            *Plan2
            *E1
        should result in: {"*Plan1": [], "*Plan2": ["*E1"]}
        This function should also return a plan map.
        """
        valid_chunk = [line for line in planner_response.splitlines()
                       if line.startswith("*Plan") or line.startswith("*E")]

        plan_to_es = dict()
        plans = dict()
        for line in valid_chunk:
            if line.startswith("*Plan"):
                plan = line.split(":", 1)[0].strip()
                plans[plan] = line.split(":", 1)[1].strip()
                plan_to_es[plan] = []
            elif line.startswith("*E"):
                plan_to_es[plan].append(line.split(":", 1)[0].strip())

        return plan_to_es, plans

    def _parse_planner_evidences(self, planner_response: str) -> (dict[str, str], List[List[str]]):
        """
        Parse planner output. This should return a mapping from *E to tool call.
        It should also identify the level of each *E in dependency map.
        Example:
            {"*E1": "Tool1", "*E2": "Tool2", "*E3": "Tool3", "*E4": "Tool4"}, [[*E1, *E2], [*E3, *E4]]
        """
        evidences, dependence = dict(), dict()
        num = 0
        for line in planner_response.splitlines():
            if line.startswith("*E") and line[2].isdigit():
                e, tool_call = line.split(":", 1)
                e, tool_call = e.strip(), tool_call.strip()
                if len(e) == 3:
                    dependence[e] = []
                    num += 1
                    evidences[e] = tool_call
                    for var in re.findall(r"\*E\d+", tool_call):
                        if var in evidences:
                            dependence[e].append(var)
                else:
                    evidences[e] = "No evidence found"
        level = []
        while num > 0:
            level.append([])
            for i in dependence:
                if dependence[i] is None:
                    continue
                if len(dependence[i]) == 0:
                    level[-1].append(i)
                    num -= 1
                    for j in dependence:
                        if j is not None and i in dependence[j]:
                            dependence[j].remove(i)
                            if len(dependence[j]) == 0:
                                dependence[j] = None

        return evidences, level

    def _get_worker_evidence(self, planner_evidences, evidences_level, output=BaseOutput()):
        worker_evidences = dict()
        for level in evidences_level:
            # TODO: Run simultaneously
            for e in level:
                tool_call = planner_evidences[e]
                if "[" not in tool_call:
                    worker_evidences[e] = tool_call
                    continue
                tool, tool_input = tool_call.split("[", 1)
                tool_input = tool_input[:-1]
                # find variables in input and replace with previous evidences
                for var in re.findall(r"\*E\d+", tool_input):
                    if var in worker_evidences:
                        tool_input = tool_input.replace(var, "[" + worker_evidences.get(var, "") + "]")
                try:
                    worker_evidences[e] = get_plugin_response_content(self._find_plugin(tool).run(tool_input))
                except:
                    worker_evidences[e] = "No evidence found."
                finally:
                    output.panel_print(worker_evidences[e], f"[green] Function Response of [blue]{tool}: ")
        return worker_evidences

    def _find_plugin(self, name: str):
        for p in self.plugins:
            if p.name == name:
                return p

    def run(self, instruction: str) -> AgentOutput:
        logging.info(f"Running {self.name + ':' + self.version} with instruction: {instruction}")
        total_cost = 0.0
        total_token = 0

        planner_llm = self._get_llms()["Planner"]
        solver_llm = self._get_llms()["Solver"]

        planner = Planner(model=planner_llm,
                          workers=self.plugins,
                          prompt_template=self.prompt_template.get("Planner", None),
                          examples=self.examples.get("Planner", None))
        solver = Solver(model=solver_llm,
                        prompt_template=self.prompt_template.get("Solver", None),
                        examples=self.examples.get("Solver", None))

        # Plan
        planner_output = planner.run(instruction)
        total_cost += calculate_cost(planner_llm.model_name, planner_output.prompt_token,
                                     planner_output.completion_token)
        total_token += planner_output.prompt_token + planner_output.completion_token
        plan_to_es, plans = self._parse_plan_map(planner_output.content)
        planner_evidences, evidence_level = self._parse_planner_evidences(planner_output.content)

        # Work
        worker_evidences = self._get_worker_evidence(planner_evidences, evidence_level)
        worker_log = ""
        for plan in plan_to_es:
            worker_log += f"{plan}: {plans[plan]}\n"
            for e in plan_to_es[plan]:
                worker_log += f"{e}: {worker_evidences[e]}\n"

        # Solve
        solver_output = solver.run(instruction, worker_log)
        total_cost += calculate_cost(solver_llm.model_name, solver_output.prompt_token,
                                     solver_output.completion_token)
        total_token += solver_output.prompt_token + solver_output.completion_token

        return AgentOutput(output=solver_output.content, cost=total_cost, token_usage=total_token)

    def stream(self, instruction: str, output: Optional[BaseOutput] = None):
        if output is None:
            output = BaseOutput()
        output.update_status(f"{self.name} is initializing...")
        planner_llm = self._get_llms()["Planner"]
        solver_llm = self._get_llms()["Solver"]
        planner = Planner(model=planner_llm,
                          workers=self.plugins,
                          prompt_template=self.prompt_template.get("Planner", None),
                          examples=self.examples.get("Planner", None))
        solver = Solver(model=solver_llm,
                        prompt_template=self.prompt_template.get("Solver", None),
                        examples=self.examples.get("Solver", None))
        output.done()

        output.thinking(f"{self.name}'s Planner is thinking...")
        response = planner.stream(instruction)
        output.done()
        output.print(f"[blue]{self.name}: ")
        planner_output = ""
        for i in response:
            planner_output += i
            if not i:
                continue
            output.panel_print(i + '\n' if i[-1] == '\n' else i, f"{self.name}'s Planner: ", True)
        output.clear()
        plan_to_es, plans = self._parse_plan_map(planner_output)
        planner_evidences, evidence_level = self._parse_planner_evidences(planner_output)

        worker_evidences = self._get_worker_evidence(planner_evidences, evidence_level, output=output)
        worker_log = ""
        for plan in plan_to_es:
            worker_log += f"{plan}: {plans[plan]}\n"
            for e in plan_to_es[plan]:
                worker_log += f"{e}: {worker_evidences[e]}\n"

        output.thinking(f"{self.name}'s Solver is thinking...")
        response = solver.stream(instruction, worker_log)
        output.done()
        solver_output = ""
        for i in response:
            solver_output += i
            if not i:
                continue
            output.panel_print(i + '\n' if i[-1] == '\n' else i, f"{self.name}'s Solver: ", True)
        output.clear()
