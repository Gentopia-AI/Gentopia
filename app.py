from pathlib import Path

import streamlit as st
from streamlit_chat import message
import os

from gentopia.assembler.agent_assembler import AgentAssembler


@st.cache_resource
def get_configs(dir: Path):
    assert dir.is_dir()
    configs = []
    for file in dir.iterdir():
        if file.is_file() and file.suffix == '.yaml':
            configs.append(file)
    return configs


def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]


@st.cache_resource
def get_agent_selected(agent):
    if 'agent' not in st.session_state or st.session_state['agent'].name != agent:
        assembler = AgentAssembler(file=agent)
        st.session_state['agent'] = assembler.get_agent()
    return st.session_state['agent']


agent_selected = st.sidebar.selectbox(
    'Select agent',
    get_configs(Path('configs'))
)
agent = get_agent_selected(agent_selected)

st.session_state.setdefault(
    'output',
    [f'Hello, I am a bot named {agent_selected}.']
)
st.session_state.setdefault(
    'user',
    []
)

st.title("Chat placeholder")

chat_placeholder = st.empty()

with chat_placeholder.container():
    index = 0
    while index < len(st.session_state['output']) or index < len(st.session_state['user']):
        if index < len(st.session_state['output']):
            message(st.session_state['output'][index], key=f"{index}_agent")
        if index < len(st.session_state['user']):
            message(
                st.session_state['user'][index],
                key=f"{index}",
                is_user=True,
                allow_html=True,
                is_table=False
            )
        index += 1

    st.button("Clear message", on_click=on_btn_click)

with st.container():
    text_element = st.empty()


    def on_input_change():
        user_input = st.session_state.user_input
        if not user_input:
            return
        st.session_state.user.append(user_input)
        ans = ""
        l = 0
        for i in agent.llm.stream_chat_completion(user_input):
            if not i.content:
                continue
            if i.content.startswith("[") or len(i.content) + l >= 80:
                break
            #     ans += "\n\n"
            #     l = 0
            l += len(i.content)
            ans += i.content
            text_element.text(ans)

        st.session_state.output.append(ans)
        text_element.text_area("")


    st.text_input("User Input:", on_change=on_input_change, key="user_input")
