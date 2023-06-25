from typing import AnyStr


from transformers import pipeline, Pipeline
from .basetool import *


class Summarize(BaseTool):
    """docstring for Summarize"""
    name = "summarize"
    description = "A tool to summarize a doc, use it when you think the response will too long. " \


    args_schema: Optional[Type[BaseModel]] = create_model("SummarizeArgs", text=(str, ...))

    summarizer: Pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

    def _run(self, text: AnyStr) -> AnyStr:
        result = self.summarizer(text, max_length=min(10240, int(len(text) / 1.5) + 1), do_sample=False)
        return result[0]['summary_text']

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
