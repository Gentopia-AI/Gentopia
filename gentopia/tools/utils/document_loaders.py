from abc import abstractmethod, ABC
from typing import Optional, List, Iterator, Sequence, Any, Callable

from gentopia.tools.utils import Document

class TextLoader:
    """Load text files."""

    def __init__(self, file_path: str, encoding: Optional[str] = None):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load from file path."""
        with open(self.file_path, encoding=self.encoding) as f:
            text = f.read()
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]