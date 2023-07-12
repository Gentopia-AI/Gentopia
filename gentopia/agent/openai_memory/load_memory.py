from typing import AnyStr

from gentopia.memory.api import MemoryWrapper
from gentopia.tools.basetool import *

# This tools is only used in openai_memory agent.
# DO NOT use this tool in other agent.
class LoadMemory(BaseTool):
    name = "load_memory"
    description = "A tool to recall the history of conversations. If you find that you do not have some information you need, you can invoke this tool with the related query string to get more information."

    args_schema: Optional[Type[BaseModel]] = create_model("LoadMemoryArgs", text=(str, ...))
    memory: MemoryWrapper

    def _run(self, text: AnyStr) -> AnyStr:
        return self.memory.load_history(text)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
