from typing import AnyStr

from langchain import OpenAI, LLMMathChain

from .basetool import *

class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="A mathematical expression.")

class Calculator(BaseTool):
    """docstring for Calculator"""
    name = "calculator"
    description = "A calculator that can compute arithmetic expressions. Useful when you need to perform " \
                  "math calculations. Input should be a mathematical expression"
    args_schema: Optional[Type[BaseModel]] = CalculatorArgs

    def _run(self, expression: AnyStr) -> Any:
        llm = OpenAI(temperature=0)
        tool = LLMMathChain(llm=llm, verbose=self.verbose)
        response = tool(expression)
        evidence = response["answer"].replace("Answer:", "").strip()
        return evidence

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError




