import logging
import os
import re
from typing import List, Dict, Union, Optional

from langchain import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
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
        Parse planner output. It should be an n-to-n mapping from Plans to #Es.
        This is because sometimes LLM cannot follow the strict output format.
        Example:
            #Plan1
            #E1
            #E2
        should result in: {"#Plan1": ["#E1", "#E2"]}
        Or:
            #Plan1
            #Plan2
            #E1
        should result in: {"#Plan1": [], "#Plan2": ["#E1"]}
        This function should also return a plan map.
        """
        valid_chunk = [line for line in planner_response.splitlines()
                       if line.startswith("#Plan") or line.startswith("#E")]

        plan_to_es = dict()
        plans = dict()
        for line in valid_chunk:
            if line.startswith("#Plan"):
                plan = line.split(":", 1)[0].strip()
                plans[plan] = line.split(":", 1)[1].strip()
                plan_to_es[plan] = []
            elif line.startswith("#E"):
                plan_to_es[plan].append(line.split(":", 1)[0].strip())

        return plan_to_es, plans

    def _parse_planner_evidences(self, planner_response: str) -> (dict[str, str], List[List[str]]):
        """
        Parse planner output. This should return a mapping from #E to tool call.
        It should also identify the level of each #E in dependency map.
        Example:
            {"#E1": "Tool1", "#E2": "Tool2", "#E3": "Tool3", "#E4": "Tool4"}, [[#E1, #E2], [#E3, #E4]]
        """
        evidences, dependence = dict(), dict()
        for line in planner_response.splitlines():
            if line.startswith("#E") and line[2].isdigit():
                e, tool_call = line.split(":", 1)
                e, tool_call = e.strip(), tool_call.strip()
                if len(e) == 3:
                    dependence[e] = []
                    evidences[e] = tool_call
                    for var in re.findall(r"#E\d+", tool_call):
                        if var in evidences:
                            dependence[e].append(var)
                else:
                    evidences[e] = "No evidence found"
        level = []
        while dependence:
            select = [i for i in dependence if not dependence[i]]
            if len(select) == 0:
                raise ValueError("Circular dependency detected.")
            level.append(select)
            for item in select:
                dependence.pop(item)
            for item in dependence:
                for i in select:
                    if i in dependence[item]:
                        dependence[item].remove(i)

        return evidences, level


    def _run_plugin(self, e, planner_evidences, worker_evidences, output=BaseOutput()):
        result = dict(e=e, plugin_cost=0, plugin_token=0, evidence="")
        tool_call = planner_evidences[e]
        if "[" not in tool_call:
            result['evidence'] = tool_call
        else:
            tool, tool_input = tool_call.split("[", 1)
            tool_input = tool_input[:-1]
            # find variables in input and replace with previous evidences
            for var in re.findall(r"#E\d+", tool_input):
                if var in worker_evidences:
                    tool_input = tool_input.replace(var, "[" + worker_evidences.get(var, "") + "]")
            try:
                tool_response = self._find_plugin(tool).run(tool_input)
                # cumulate agent-as-plugin costs and tokens.
                if isinstance(tool_response, AgentOutput):
                    result['plugin_cost'] = tool_response.cost
                    result['plugin_token'] = tool_response.token_usage
                result['evidence'] = get_plugin_response_content(tool_response)
            except:
                result['evidence'] = "No evidence found."
            finally:
                output.panel_print(result['evidence'], f"[green] Function Response of [blue]{tool}: ")
        return result


    def _get_worker_evidence(self, planner_evidences, evidences_level, output=BaseOutput()):
        worker_evidences = dict()
        plugin_cost, plugin_token = 0.0, 0.0
        with ThreadPoolExecutor(max_workers=2) as pool:
            for level in evidences_level:
                results = []
                for e in level:
                    results.append(pool.submit(self._run_plugin, e, planner_evidences, worker_evidences, output))
                if len(results) > 1:
                    output.update_status(f"Running tasks {level} in parallel.")
                else:
                    output.update_status(f"Running task {level[0]}.")
                for r in results:
                    resp = r.result()
                    plugin_cost += resp['plugin_cost']
                    plugin_token += resp['plugin_token']
                    worker_evidences[resp['e']] = resp['evidence']
                output.done()

        return worker_evidences, plugin_cost, plugin_token

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
        worker_evidences, plugin_cost, plugin_token = self._get_worker_evidence(planner_evidences, evidence_level)
        worker_log = ""
        for plan in plan_to_es:
            worker_log += f"{plan}: {plans[plan]}\n"
            for e in plan_to_es[plan]:
                worker_log += f"{e}: {worker_evidences[e]}\n"

        # Solve
        solver_output = solver.run(instruction, worker_log)
        total_cost += calculate_cost(solver_llm.model_name, solver_output.prompt_token,
                                     solver_output.completion_token) + plugin_cost
        total_token += solver_output.prompt_token + solver_output.completion_token + plugin_token

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

        worker_evidences, _, _ = self._get_worker_evidence(planner_evidences, evidence_level, output=output)
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
