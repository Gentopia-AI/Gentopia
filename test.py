from langchain.prompts import HumanMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.schema import SystemMessage

#
# x = Config.load("gentopia/agent/agent.yaml")
# print(x)
#
# from gentopia.config.agent_config import AgentConfig
#
# x = AgentConfig("gentopia/agent/agent.yaml")
# a = x.get_agent()
# print(a)
# print(a.tools.tools['Wikipedia'].description)
# print(a.tools.generate_worker_prompt())
# print(x)

if __name__ == '__main__':
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    input_variables = ["input", "agent_scratchpad"]
    x = ChatPromptTemplate(input_variables=input_variables, messages=messages)
    print(x.to_messages())
