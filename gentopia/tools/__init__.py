from .basetool import BaseTool
from .calculator import Calculator
from .doc_store_look_up import DocStoreLookUp
# from .dummy_tool import DummyTool
from .google_search import GoogleSearch
from .llm import LLM
from .search_doc import SearchDoc
from .wikipedia import Wikipedia
from .wolfram_alpha import WolframAlpha
from .zip_code import ZipCodeRetriever


def load_tools(name: str) -> BaseTool:
    if name == "calculator":
        return Calculator
    elif name == "doc_store_look_up":
        return DocStoreLookUp
    elif name == "google_search":
        return GoogleSearch
    elif name == "llm":
        return LLM
    elif name == "search_doc":
        return SearchDoc
    elif name == "wikipedia":
        return Wikipedia
    elif name == "wolfram_alpha":
        return WolframAlpha
    elif name == "zip_code":
        return ZipCodeRetriever
    else:
        raise NotImplementedError
