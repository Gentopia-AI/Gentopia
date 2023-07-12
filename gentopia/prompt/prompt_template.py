"""Prompt schema definition."""
from __future__ import annotations

from string import Formatter
from typing import Any, Dict, List
from pydantic import BaseModel

from pydantic import Extra, root_validator


class PromptTemplate(BaseModel):
    """A simple yet sufficient prompt schema with a placeholder for f-string.

    Example:
        .. code-block:: python

            from gentopia import PromptTemplate
            prompt = PromptTemplate(input_variables=["foo"], template="Say {foo}")
    """

    input_variables: List[str]
    template: str
    validate_template: bool = True
    skip_on_failure = True

    class Config:
        extra = Extra.forbid

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that template and input variables are consistent."""
        if values["validate_template"]:
            try:
                dummy_input = {var: "" for var in values["input_variables"]}
                Formatter().format(values["template"], **dummy_input)
            except KeyError as e:
                raise ValueError(
                    "Invalid prompt schema; check for mismatched or missing input parameters. "
                    + str(e)
                )
        return values


# For backwards compatibility.
Prompt = PromptTemplate
