from .basetool import BaseTool
from .calculator import Calculator
from .google_search import GoogleSearch
from .llm import LLM
from .search_doc import SearchDoc
from .wikipedia import Wikipedia
from .wolfram_alpha import WolframAlpha
from .web_page import WebPage
# from .summarize import Summarize

def load_tools(name: str) -> BaseTool:
    if name == "calculator":
        return Calculator
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
    elif name == "web_page":
        return WebPage
    # elif name == "summarize":
    #     return Summarize
    else:
        raise NotImplementedError
