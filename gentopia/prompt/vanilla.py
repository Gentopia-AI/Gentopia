from langchain import PromptTemplate

ZeroShotVanillaPrompt = PromptTemplate(
    input_variables=["instruction"],
    template="""{instruction}"""
)

FewShotVanillaPrompt = PromptTemplate(
    input_variables=["instruction", "fewshot"],
    template="""{fewshot}
    
{instruction}"""
)
