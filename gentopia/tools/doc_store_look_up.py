from typing import AnyStr

from .basetool import *


class DocStoreLookUp(BaseTool):
    """docstring for DocStoreLookUp"""
    name = "doc_store_look_up"
    description = "Worker that search the direct sentence in current Wikipedia result page. Useful when you " \
                  "need to find information about a specific keyword from a existing Wikipedia search " \
                  "result. Input should be a search keyword."
    args_schema = create_model("DocStoreLookUpArgs", query=(str, ...))

    def __init__(self, doc_store=None):
        super().__init__()
        self.doc_store = doc_store

    def _run(self, query: AnyStr) -> AnyStr:
        assert self.doc_store is not None, "doc_store is not set"
        tool = self.doc_store
        evidence = tool.search(query)
        return evidence

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
