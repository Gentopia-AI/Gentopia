import streamlit as st
from pydantic import BaseModel
import streamlit_pydantic as sp

from gentopia.agent.rewoo.nodes.Planner import Planner
from gentopia.assembler.agent_assembler import AgentAssembler
from gentopia.assembler.config import Config
from gentopia.llm import HuggingfaceLLMClient
from gentopia.llm.base_llm import BaseLLM
from gentopia.manager.local_llm_manager import LocalLLMManager
from gentopia.model.param_model import HuggingfaceParamModel
import multiprocessing as mp
import streamlit as st


@st.cache_resource
def get_manager():
    return LocalLLMManager()


@st.cache_resource
def get_agent(agent):
    # Fetch data from URL here, and then clean it up.
    assembler = AgentAssembler(file=f'config/{agent}.yaml')
    assembler.manager = get_manager()
    return assembler.get_agent()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    ans = ""
    with st.sidebar:
        agent_name = st.radio(
            "Choose an agent: ",
            ("Alice", "Bob", "Carol")
        )
    if agent_name:
        st.title(f"{agent_name} Config")
        if agent_name == "Alice":
            config = Config.load('config/alice.yaml')
            st.json(config)
        elif agent_name == "Bob":
            config = Config.load('config/bob.yaml')
            st.json(config)
        elif agent_name == "Carol":
            config = Config.load('config/carol.yaml')
            st.json(config)
        st.title(f"{agent_name} Chat")
        _agent = get_agent(agent_name)

        inp = st.text_input("User Input:")
        start = st.button("Start")
        if start:
            if inp:
                l = 0
                inp = ans + f"[User]: {inp.strip()}\n"
                ans = ""
                text_element = st.empty()
                if agent_name == "Carol":
                    agent = _agent._get_llms()["Planner"]

                    planner = Planner(model=agent,
                                      workers=_agent.plugins,
                                      prompt_template=_agent.prompt_template.get("Planner", None),
                                      examples=_agent.examples.get("Planner", None))
                    inp = planner._compose_prompt(inp)
                else:
                    agent = _agent.llm
                for i in agent.stream_chat_completion(inp):
                    if not i.content:
                        continue
                    if i.content.startswith("[") or len(i.content) + l >= 80:
                        ans += "\n\n"
                        l = 0
                    l += len(i.content)
                    ans += i.content
                    text_element.text(ans)
                st.divider()
