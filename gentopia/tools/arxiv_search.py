from typing import AnyStr, Any
import arxiv
from gentopia.tools.basetool import *


class ArxivSearch(BaseTool):
    """Tool that adds the capability to query Axiv search api"""

    name = "ArxivSearch"
    description = (
        "Search information from Arxiv.org "
        "Useful for when you need to answer questions about Physics, Mathematics, "
        "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
        "Electrical Engineering, and Economics "
        "from scientific articles on arxiv.org. "
        "Input should be a search query."
    )
    args_schema: Optional[Type[BaseModel]] = create_model("ArxivSearchArgs", query=(str, ...))

    def _run(self, query: AnyStr) -> AnyStr:
        # arxiv_exceptions: Any  # :meta private:
        top_k_results: int = 3
        ARXIV_MAX_QUERY_LENGTH = 300
        doc_content_chars_max: int = 4000
        try:
            results = arxiv.Search(
                query[: ARXIV_MAX_QUERY_LENGTH], max_results=top_k_results
            ).results()
        # except arxiv_exceptions as ex:
        except Exception as ex:
            return f"Arxiv exception: {ex}"
        docs = [
            f"Published: {result.updated.date()}\nTitle: {result.title}\n"
            f"Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Summary: {result.summary}"
            for result in results
        ]
        if docs:
            return "\n\n".join(docs)[: doc_content_chars_max]
        else:
            return "No good Arxiv Result was found"

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


if __name__ == "__main__":
    ans = ArxivSearch()._run("Attention for transformer")
    print(ans)
