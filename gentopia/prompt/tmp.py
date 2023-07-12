from gentopia.prompt import PromptTemplate

SummaryVanillaPrompt = PromptTemplate(
    input_variables=["instruction"],
    template="""
    Summarize the following text:
    {instruction}
    """
)