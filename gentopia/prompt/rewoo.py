from langchain import PromptTemplate

ZeroShotPlannerPrompt = PromptTemplate(
    input_variables=["tool_description", "task"],
    template="""You are an AI agent who makes step-by-step plans to solve a problem under the help of external tools. 
For each step, make one plan followed by one tool-call, which will be executed later to retrieve evidence for that step.
You should store each evidence into a distinct variable #E1, #E2, #E3 ... that can be referred to in later tool-call inputs.    

##Available Tools##
{tool_description}

##Output Format (Replace '<...>')##
*Plan1: <describe your plan here>
*E1: <toolname>[<input>] (eg. Search[What is Python])
*Plan2: <describe next plan>
*E2: <toolname>[<input, you can use #E1 to represent its expected output>]
And so on...
  
##Your Task##
{task}

##Now Begin##
"""
)

FewShotPlannerPrompt = PromptTemplate(
    input_variables=["tool_description", "fewshot", "task"],
    template="""You are an AI agent who makes step-by-step plans to solve a problem under the help of external tools. 
For each step, make one plan followed by one tool-call, which will be executed later to retrieve evidence for that step.
You should store each evidence into a distinct variable #E1, #E2, #E3 ... that can be referred to in later tool-call inputs.    

##Available Tools##
{tool_description}

##Output Format (Replace '<...>')##
*Plan1: <describe your plan here>
*E1: <toolname>[<input>] 
*Plan2: <describe next plan>
*E2: <toolname>[<input, you can use #E1 to represent its expected output>]
And so on...

##Examples##
{fewshot}

##Your Task##
{task}

##Now Begin##
"""
)

ZeroShotSolverPrompt = PromptTemplate(
    input_variables=["plan_evidence", "task"],
    template="""You are an AI agent who solves a problem with my assistance. I will provide step-by-step plans(*Plan) and evidences(*E) that could be helpful.
Your task is to briefly summarize each step, then make a short final conclusion for your task.

##My Plans and Evidences##
{plan_evidence}

##Example Output##
First, I <did something> , and I think <...>; Second, I <...>, and I think <...>; ....
So, <your conclusion>.

##Your Task##
{task}

##Now Begin##
"""
)

FewShotSolverPrompt = PromptTemplate(
    input_variables=["plan_evidence", "fewshot", "task"],
    template="""You are an AI agent who solves a problem with my assistance. I will provide step-by-step plans and evidences that could be helpful.
Your task is to briefly summarize each step, then make a short final conclusion for your task.

##My Plans and Evidences##
{plan_evidence}

##Example Output##
First, I <did something> , and I think <...>; Second, I <...>, and I think <...>; ....
So, <your conclusion>.

##Example##
{fewshot}

##Your Task##
{task}

##Now Begin##
"""
)