from langchain import PromptTemplate

SummaryVanillaPrompt = PromptTemplate(
    input_variables=["instruction"],
    template="""
    Summarize the following text:
    {instruction}
    """
)