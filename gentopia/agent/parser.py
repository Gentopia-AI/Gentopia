import re


def parse_plans(response):
    return [line for line in response.splitlines() if line.startswith("Plan:")]


def parse_planner_evidences(response):
    evidences = dict()
    num = 0
    dependence = dict()
    for line in response.splitlines():
        if line.startswith("#") and line[1] == "E" and line[2].isdigit():
            e, tool_call = line.split("=", 1)
            e, tool_call = e.strip(), tool_call.strip()
            if len(e) == 3:
                dependence[e] = []
                num += 1
                evidences[e] = tool_call
                for var in re.findall(r"#E\d+", tool_call):
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
