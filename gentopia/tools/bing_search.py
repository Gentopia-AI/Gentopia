import os
from enum import Enum
from typing import AnyStr

import requests
from bs4 import BeautifulSoup

from .basetool import *


class BingAPI:
    """
    A class for performing searches on the Bing search engine.

    Attributes
    ----------
    bing_api : BingAPI
        The Bing API to use for performing searches.

    Methods
    -------
    __init__(self, subscription_key: str) -> None:
        Initialize the BingSearch instance with the given subscription key.
    search_top3(self, key_words: str) -> List[str]:
        Perform a search on the Bing search engine with the given keywords and return the top 3 search results.
    load_page_index(self, idx: int) -> str:
        Load the detailed page of the search result at the given index.
    """
    def __init__(self, subscription_key : str) -> None:
        """
        Initialize the BingSearch instance with the given subscription key.

        Parameters
        ----------
        subscription_key : str
            The subscription key to use for the Bing API.
        """
        self._headers = {
            'Ocp-Apim-Subscription-Key': subscription_key
        }
        self._endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self._mkt = 'en-US'
    
    def search(self, key_words : str, max_retry : int = 3):
        for _ in range(max_retry):
            try:
                result = requests.get(self._endpoint, headers=self._headers, params={'q': key_words, 'mkt': self._mkt }, timeout=10)
            except Exception:
                # failed, retry
                continue

            if result.status_code == 200:
                result = result.json()
                # search result returned here
                return result
            else:
                # failed, retry
                continue
        raise RuntimeError("Failed to access Bing Search API.")
    
    def load_page(self, url : str, max_retry : int = 3) -> Tuple[bool, str]:
        for _ in range(max_retry):
            try:
                res = requests.get(url, timeout=15)
                if res.status_code == 200:
                    res.raise_for_status()
                else:
                    raise RuntimeError("Failed to load page, code {}".format(res.status_code))
            except Exception:
                # failed, retry
                res = None
                continue
            res.encoding = res.apparent_encoding
            content = res.text
            break
        if res is None:
            return False, "Timeout for loading this page, Please try to load another one or search again."
        try:
            soup = BeautifulSoup(content, 'html.parser')
            paragraphs = soup.find_all('p')
            page_detail = ""
            for p in paragraphs:
                text = p.get_text().strip()
                page_detail += text
            return True, page_detail
        except Exception:
            return False, "Timeout for loading this page, Please try to load another one or search again."


class CONTENT_TYPE(Enum):
    SEARCH_RESULT = 0
    RESULT_TARGET_PAGE = 1


class ContentItem:
    def __init__(self, type: CONTENT_TYPE, data):
        self.type = type
        self.data = data


class SessionData:
    content = []
    curResultChunk = 0
    

# data = SessionData()
# 
# 
# def build_tool(config) -> Tool:
#     tool = Tool(
#         "Bing_search",
#         "Bing_search",
#         name_for_model="Bing_search",
#         name_for_human="Bing_search",
#         description_for_model="""Perform Search on Bing Search engine.
# Use search_top3(key: str) to get top 3 search results after input the key to search.
# Use load_page_index(idx: int) to load the detailed page of the search result.""",
#         description_for_human="Bing search API for browsing the internet and search for results.",
#         logo_url="https://your-app-url.com/.well-known/logo.png",
#         contact_email="hello@contact.com",
#         legal_info_url="hello@legal.com"
#     )
# 
#     if "debug" in config and config["debug"]:
#         bing_api = config["bing_api"]
#     else:
#         bing_api = BingAPI(config["subscription_key"])
# 
#     @tool.get("/search_top3")
#     def search_top3(key_words: str) -> str:
#         """Search key words, return top 3 search results.
#         """
#         top3 = search_all(key_words)[:3]
#         output = ""
#         for idx, item in enumerate(top3):
#             output += "page: " + str(idx+1) + "\n"
#             output += "title: " + item['name'] + "\n"
#             output += "summary: " + item['snippet'] + "\n"
#         return output
# 
#     def search_all(key_words: str, data: SessionData = data) -> list:
#         """Search key_words, return a list of class SearchResult.
#         Keyword arguments:
#         key_words -- key words want to search
#         """
#         result = bing_api.search(key_words)
#         data.content = []
#         data.content.append(ContentItem(CONTENT_TYPE.SEARCH_RESULT, result))
#         data.curResultChunk = 0
#         return data.content[-1].data["webPages"]["value"]
# 
#     @tool.get("/load_page_index")
#     def load_page_index(idx: str) -> str:
#         """Load page detail of the search result indexed as 'idx', and return the content of the page.
#         """
#         idx = int(idx)
#         href, text = load_page(idx-1)
#         if len(text) > 500:
#             return text[:500]
#         else:
#             return text
# 
#     def load_page(idx : int, data: SessionData = data):
#         top = data.content[-1].data["webPages"]["value"]
#         ok, content = bing_api.load_page(top[idx]['url'])
#         if ok:
#             return top[idx]['url'], content
#         else:
#             return " ", "Timeout for loading this page, Please try to load another one or search again."
#         
#     return tool


class BingSearch(BaseTool):
    api_key: str = os.getenv("BING_API_KEY")
    search_engine: BingAPI = BingAPI(api_key)
    session_data: SessionData = SessionData()


class BingSearchTop3(BingSearch):
    """Tool that adds the cability to query Bing search API,
    return top-3 search results
    """

    name = "BingSearchTop3"
    description = "search from Bing and return top-3 results." \
                  "Input should be a search query."
    args_schema: Optional[Type[BaseModel]] = create_model("BingSearchTop3Args", query=(str, ...))

    def _run(self, query: AnyStr) -> AnyStr:
        top3 = self.search_all(query)[:3]
        output = ""
        for idx, item in enumerate(top3):
            output += "page: " + str(idx+1) + "\n"
            output += "title: " + item['name'] + "\n"
            output += "summary: " + item['snippet'] + "\n"
        return output

    def search_all(self, key_words: str) -> list:
        """Search key_words, return a list of class SearchResult.
        Keyword arguments:
        key_words -- key words want to search
        """
        result = self.search_engine.search(key_words)
        self.session_data.content = []
        self.session_data.content.append(ContentItem(CONTENT_TYPE.SEARCH_RESULT, result))
        self.session_data.curResultChunk = 0
        return self.session_data.content[-1].data["webPages"]["value"]

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class BingSearchLoadPage(BingSearch):
    """select one page from the results of BingSearchTop3 and
       check the details3
    """

    name = "BingSearchLoadPage"
    description = "Load page details of the search results from BingsearchTop3" \
                  "Input should be an index starting from 1."
    args_schema: Optional[Type[BaseModel]] = create_model("BingSearchLoadPageArgs", idx=(int, ...))

    def _run(self, idx: int) -> AnyStr:
        href, text = self.load_page(idx-1)
        if len(text) > 500:
            return text[:500]
        else:
            return text

    def load_page(self, idx : int) -> AnyStr:
        top = self.session_data.content[-1].data["webPages"]["value"]
        ok, content = self.search_engine.load_page(top[idx]['url'])
        if ok:
            return top[idx]['url'], content
        else:
            return " ", "Timeout for loading this page, Please try to load another one or search again."

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


        
if __name__ == "__main__":
    ans1 = BingSearchTop3()._run("What is the weather today?")
    print(ans1)
    ans2 = BingSearchLoadPage()._run(1, top3)
    print(ans2)
