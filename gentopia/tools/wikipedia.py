from typing import AnyStr

from langchain import Wikipedia as Wiki
from langchain.agents.react.base import DocstoreExplorer

from .basetool import *


class Wikipedia(BaseTool):
    """Tool that adds the capability to query the Wikipedia API."""

    name = "Wikipedia"
    description = "Worker that search for similar page contents from Wikipedia. Useful when you need to " \
                  "get holistic knowledge about people, places, companies, historical events, " \
                  "or other subjects. The response are long and might contain some irrelevant information. " \
                  "Input should be a search query."
    args_schema: Optional[Type[BaseModel]] = create_model("WikipediaArgs", query=(str, ...))
    doc_store: Any = None

    def _run(self, query: AnyStr) -> AnyStr:
        if not self.doc_store:
            self.doc_store = DocstoreExplorer(Wiki())
        tool = self.doc_store
        evidence = tool.search(query)
        return evidence

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
