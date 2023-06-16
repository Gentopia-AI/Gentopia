from typing import AnyStr

from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

from .basetool import *


class WikipediaSearch(BaseTool):
    """Tool that adds the capability to query the Wikipedia API."""

    name = "Wikipedia"
    description = "Worker that search for similar page contents from Wikipedia. Useful when you need to " \
                  "get holistic knowledge about people, places, companies, historical events, " \
                  "or other subjects. The response are long and might contain some irrelevant information. " \
                  "Input should be a search query."
    args_schema = create_model("WikipediaArgs", query=(str, ...))

    def __init__(self, doc_store=None):
        super().__init__()
        self.doc_store = doc_store

    def _run(self, query: AnyStr) -> AnyStr:
        if not self.doc_store:
            self.doc_store = DocstoreExplorer(Wikipedia())
        tool = self.doc_store
        evidence = tool.search(query)
        return evidence

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
