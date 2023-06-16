from typing import AnyStr

from langchain import SerpAPIWrapper

from .basetool import *


class GoogleSearch(BaseTool):
    """Tool that adds the capability to query the Google search API."""

    name = "GoogleSearch"
    description = "Worker that searches results from Google. Useful when you need to find short " \
                  "and succinct answers about a specific topic. Input should be a search query."

    args_schema = create_model("GoogleSearchArgs", query=(str, ...))

    def _run(self, query: AnyStr) -> AnyStr:
        tool = SerpAPIWrapper()
        return tool.run(query)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
