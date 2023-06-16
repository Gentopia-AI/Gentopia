from typing import AnyStr

from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

from .basetool import *


class SearchDoc(BaseTool):
    name = "SearchDoc"
    args_schema = create_model("SearchDocArgs", query=(str, ...))

    def __init__(self, doc_name, doc_path):
        super().__init__()
        self.doc_path = doc_path
        self.description = f"A vector store that searches for similar and related content in document: {doc_name}. " \
                           f"The result is a huge chunk of text related to your search but can also " \
                           f"contain irrelevant info. Input should be a search query."

    def _run(self, query: AnyStr) -> AnyStr:
        loader = TextLoader(self.doc_path)
        vector_store = VectorstoreIndexCreator().from_loaders([loader]).vectorstore
        evidence = vector_store.similarity_search(query, k=1)[0].page_content
        return evidence

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
