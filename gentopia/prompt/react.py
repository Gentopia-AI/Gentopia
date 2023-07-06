from langchain import PromptTemplate

ZeroShotReactPrompt = PromptTemplate(
    input_variables=["instruction", "agent_scratchpad", "tool_names", "tool_description"],
    template="""Answer the following questions as best you can. You have access to the following tools:
{tool_description}.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
#Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! After each Action Input.

Question: {instruction}
Thought:{agent_scratchpad}
    """
)