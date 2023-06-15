from typing import AnyStr

from langchain import WolframAlphaAPIWrapper

from .basetool import *


class CustomWolframAlphaAPITool(WolframAlphaAPIWrapper):
    def __init__(self):
        super().__init__()

    def run(self, query: str) -> str:
        """Run query through WolframAlpha and parse result."""
        res = self.wolfram_client.query(query)

        try:
            answer = next(res.results).text
        except StopIteration:
            return "Wolfram Alpha wasn't able to answer it"

        if answer is None or answer == "":
            return "No good Wolfram Alpha Result was found"
        else:
            return f"Answer: {answer}"


class WolframAlphaWorker(BaseTool):
    name = "wolfram_alpha"
    description = "A WolframAlpha search engine. Useful when you need to solve a complicated Mathematical or " \
                  "Algebraic equation. Input should be an equation or function."
    args_schema = create_model("WolframAlphaArgs", query=(str, ...))

    def _run(self, query: AnyStr) -> AnyStr:
        tool = CustomWolframAlphaAPITool()
        evidence = tool.run(query).replace("Answer:", "").strip()
        return evidence

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
