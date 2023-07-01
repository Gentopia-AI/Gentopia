from pathlib import Path
from typing import AnyStr, Any
import arxiv
from basetool import *


class WriteFileArgs(BaseModel):
    file_path: str = Field(..., description="the path to write the file")
    text: str = Field(..., description="the string to store")


class WriteFile(BaseTool):
    """write file to disk"""

    name = "WriteFile"
    description = (
        "Write strings to a file in hardisk"
        "Useful for when you need to store some results "
    )
    args_schema: Optional[Type[BaseModel]] = WriteFileArgs

    def _run(self, file_path, text) -> AnyStr:
        write_path = (
            Path(file_path)
        )
        try:
            write_path.parent.mkdir(exist_ok=True, parents=False)
            with write_path.open("w", encoding="utf-8") as f:
                f.write(text)
            return f"File written successfully to {file_path}."
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class ReadFileArgs(BaseModel):
    file_path: str = Field(..., description="the path to read the file")


class ReadFile(BaseTool):
    """read file from disk"""

    name = "ReadFile"
    description = (
        "Read a file from hardisk"
    )
    args_schema: Optional[Type[BaseModel]] = ReadFileArgs

    def _run(self, file_path) -> AnyStr:
        read_path = (
            Path(file_path)
        )
        try:
            with read_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


if __name__ == "__main__":
    ans = WriteFile()._run("hello_world.text", "hello_world")
    # ans = ReadFile()._run("hello_world.text")
    print(ans)

