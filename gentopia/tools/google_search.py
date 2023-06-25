from typing import AnyStr, List

from googlesearch import search, SearchResult

from .basetool import *


class GoogleSearch(BaseTool):
    """Tool that adds the capability to query the Google search API."""

    name = "GoogleSearch"
    description = "Tool that searches results from Google. Input should be a search query."
    # \Useful when you need to find short " \
    #               "and succinct answers about a specific topic. Input should be a search query."

    args_schema: Optional[Type[BaseModel]] = create_model("GoogleSearchArgs", query=(str, ...))

    def _run(self, query: AnyStr) -> str:

        return '\n\n'.join([str(item) for item in search(query, advanced=True)])

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
