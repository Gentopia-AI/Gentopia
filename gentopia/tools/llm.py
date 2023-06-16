from typing import AnyStr

from langchain import OpenAI, PromptTemplate, LLMChain

from .basetool import *


class LLM(BaseTool):
    """docstring for LLM"""
    name = "llm"
    description = "A pretrained LLM like yourself. Useful when you need to act with general world " \
                  "knowledge and common sense. Prioritize it when you are confident in solving the problem " \
                  "yourself. Input can be any instruction."

    args_schema = create_model("LLMArgs", text=(str, ...))

    def _run(self, text: AnyStr) -> AnyStr:
        llm = OpenAI(temperature=0)
        prompt = PromptTemplate(template="Respond in short directly with no extra words.\n\n{request}",
                                input_variables=["request"])
        tool = LLMChain(prompt=prompt, llm=llm, verbose=False)
        response = tool(input)
        evidence = response["text"].strip("\n")
        return evidence

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
