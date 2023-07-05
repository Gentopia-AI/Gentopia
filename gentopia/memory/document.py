from gentopia.memory.serializable import Serializable
from pydantic import Field


class Document(Serializable):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)
