"""Base implementation for tools or skills. """
from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature, iscoroutinefunction
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Type, Union

from pydantic import (
    BaseModel,
    Extra,
    Field,
    create_model,
    validate_arguments,
)
from pydantic.main import ModelMetaclass


class SchemaAnnotationError(TypeError):
    """Raised when 'args_schema' is missing or has an incorrect type annotation."""


class ToolMetaclass(ModelMetaclass):
    """Metaclass for BaseTool to ensure the provided args_schema

    doesn't silently ignore."""

    def __new__(
            cls: Type[ToolMetaclass], name: str, bases: Tuple[Type, ...], dct: dict
    ) -> ToolMetaclass:
        """Create the definition of the new tool class."""
        schema_type: Optional[Type[BaseModel]] = dct.get("args_schema")
        if schema_type is not None:
            schema_annotations = dct.get("__annotations__", {})
            args_schema_type = schema_annotations.get("args_schema", None)
            if args_schema_type is None or args_schema_type == BaseModel:
                # Throw errors for common mis-annotations.
                # TODO: Use get_args / get_origin and fully
                # specify valid annotations.
                typehint_mandate = """
class ChildTool(BaseTool):
    ...
    args_schema: Type[BaseModel] = SchemaClass
    ..."""
                raise SchemaAnnotationError(
                    f"Tool definition for {name} must include valid type annotations"
                    f" for argument 'args_schema' to behave as expected.\n"
                    f"Expected annotation of 'Type[BaseModel]'"
                    f" but got '{args_schema_type}'.\n"
                    f"Expected class looks like:\n"
                    f"{typehint_mandate}"
                )
        # Pass through to Pydantic's metaclass
        return super().__new__(cls, name, bases, dct)


def _create_subset_model(
        name: str, model: BaseModel, field_names: list
) -> Type[BaseModel]:
    """Create a pydantic model with only a subset of model's fields."""
    fields = {
        field_name: (
            model.__fields__[field_name].type_,
            model.__fields__[field_name].default,
        )
        for field_name in field_names
        if field_name in model.__fields__
    }
    return create_model(name, **fields)  # type: ignore


def get_filtered_args(
        inferred_model: Type[BaseModel],
        func: Callable,
) -> dict:
    """Get the arguments from a function's signature."""
    schema = inferred_model.schema()["properties"]
    valid_keys = signature(func).parameters
    return {k: schema[k] for k in valid_keys if k != "run_manager"}


class _SchemaConfig:
    """Configuration for the pydantic model."""

    extra = Extra.forbid
    arbitrary_types_allowed = True


def create_schema_from_function(
        model_name: str,
        func: Callable,
) -> Type[BaseModel]:
    """Create a pydantic schema from a function's signature."""
    validated = validate_arguments(func, config=_SchemaConfig)  # type: ignore
    inferred_model = validated.model  # type: ignore
    if "run_manager" in inferred_model.__fields__:
        del inferred_model.__fields__["run_manager"]
    # Pydantic adds placeholder virtual fields we need to strip
    filtered_args = get_filtered_args(inferred_model, func)
    return _create_subset_model(
        f"{model_name}Schema", inferred_model, list(filtered_args)
    )


class ToolException(Exception):
    """An optional exception that tool throws when execution error occurs.

    When this exception is thrown, the agent will not stop working,
    but will handle the exception according to the handle_tool_error
    variable of the tool, and the processing result will be returned
    to the agent as observation, and printed in red on the console.
    """

    pass


class BaseTool(ABC, BaseModel, metaclass=ToolMetaclass):
    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: str
    """Used to tell the model how/when/why to use the tool.

    You can provide few-shot examples as a part of the description.
    """
    args_schema: Optional[Type[BaseModel]] = None
    """Pydantic model class to validate and parse the tool's input arguments."""

    verbose: bool = False
    """Whether to log the tool's progress."""

    handle_tool_error: Optional[
        Union[bool, str, Callable[[ToolException], str]]
    ] = False
    """Handle the content of the ToolException thrown."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def is_single_input(self) -> bool:
        """Whether the tool only accepts a single input."""
        keys = {k for k in self.args if k != "kwargs"}
        return len(keys) == 1

    @property
    def args(self) -> dict:
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        else:
            schema = create_schema_from_function(self.name, self._run)
            return schema.schema()["properties"]

    def _parse_input(
            self,
            tool_input: Union[str, Dict],
    ) -> Union[str, Dict[str, Any]]:
        """Convert tool input to pydantic model."""
        input_args = self.args_schema
        if isinstance(tool_input, str):
            if input_args is not None:
                key_ = next(iter(input_args.__fields__.keys()))
                input_args.validate({key_: tool_input})
            return tool_input
        else:
            if input_args is not None:
                result = input_args.parse_obj(tool_input)
                return {k: v for k, v in result.dict().items() if k in tool_input}
        return tool_input

    @abstractmethod
    def _run(
            self,
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        """Call tool."""

    @abstractmethod
    async def _arun(
            self,
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        """Call the tool asynchronously."""

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        # For backwards compatibility, if run_input is a string,
        # pass as a positional argument.
        if isinstance(tool_input, str):
            return (tool_input,), {}
        else:
            return (), tool_input

    def _handle_tool_error(self, e: ToolException) -> Any:
        """Handle the content of the ToolException thrown."""
        observation = None
        if not self.handle_tool_error:
            raise e
        elif isinstance(self.handle_tool_error, bool):
            if e.args:
                observation = e.args[0]
            else:
                observation = "Tool execution error"
        elif isinstance(self.handle_tool_error, str):
            observation = self.handle_tool_error
        elif callable(self.handle_tool_error):
            observation = self.handle_tool_error(e)
        else:
            raise ValueError(
                f"Got unexpected type of `handle_tool_error`. Expected bool, str "
                f"or callable. Received: {self.handle_tool_error}"
            )
        return observation

    def run(
            self,
            tool_input: Union[str, Dict],
            verbose: Optional[bool] = None,
            **kwargs: Any,
    ) -> Any:
        """Run the tool."""
        parsed_input = self._parse_input(tool_input)
        verbose_ = verbose if not self.verbose and verbose is not None else self.verbose
        # TODO (verbose_): Add logging
        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            observation = self._run(*tool_args, **tool_kwargs)
        except ToolException as e:
            observation = self._handle_tool_error(e)
            return observation
        else:
            return observation

    async def arun(
            self,
            tool_input: Union[str, Dict],
            verbose: Optional[bool] = None,
            **kwargs: Any,
    ) -> Any:
        """Run the tool asynchronously."""
        parsed_input = self._parse_input(tool_input)
        verbose_ = verbose if not self.verbose and verbose is not None else self.verbose
        # TODO (verbose_): Add logging
        try:
            # We then call the tool on the tool input to get an observation
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            observation = await self._arun(*tool_args, **tool_kwargs)
        except ToolException as e:
            observation = self._handle_tool_error(e)
            return observation
        except (Exception, KeyboardInterrupt) as e:
            raise e
        else:
            return observation

    def __call__(self, tool_input: str) -> str:
        """Make tool callable."""
        return self.run(tool_input)


class Tool(BaseTool):
    """Tool that takes in function or coroutine directly."""

    description: str = ""
    func: Callable[..., str]
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[str]]] = None
    """The asynchronous version of the function."""

    @property
    def args(self) -> dict:
        """The tool's input arguments."""
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        # For backwards compatibility, if the function signature is ambiguous,
        # assume it takes a single string input.
        return {"tool_input": {"type": "string"}}

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        """Convert tool input to pydantic model."""
        args, kwargs = super()._to_args_and_kwargs(tool_input)
        # For backwards compatibility. The tool must be run with a single input
        all_args = list(args) + list(kwargs.values())
        if len(all_args) != 1:
            raise ValueError(
                f"Too many arguments to single-input tool {self.name}."
                f" Args: {all_args}"
            )
        return tuple(all_args), {}

    def _run(
            self,
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        """Use the tool."""
        return self.func(*args, **kwargs)

    async def _arun(
            self,
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        """Use the tool asynchronously."""
        if self.coroutine:
            return await self.coroutine(*args, **kwargs)
        raise NotImplementedError("Tool does not support async")

    def __init__(
            self, name: str, func: Callable, description: str, **kwargs: Any
    ) -> None:
        """Initialize tool."""
        super(Tool, self).__init__(
            name=name, func=func, description=description, **kwargs
        )
        if iscoroutinefunction(func):
            self.coroutine = func

    @classmethod
    def from_function(
            cls,
            func: Callable,
            name: str,
            description: str,
            args_schema: Optional[Type[BaseModel]] = None,
            **kwargs: Any,
    ) -> Tool:
        """Initialize tool from a function."""
        return cls(
            name=name,
            func=func,
            description=description,
            args_schema=args_schema,
            **kwargs,
        )


class StructuredTool(BaseTool):
    """Tool that can operate on any number of inputs."""

    description: str = ""
    args_schema: Type[BaseModel] = Field(..., description="The tool schema.")
    """The input arguments' schema."""
    func: Callable[..., Any]
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None
    """The asynchronous version of the function."""

    @property
    def args(self) -> dict:
        """The tool's input arguments."""
        return self.args_schema.schema()["properties"]

    def _run(
            self,
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        """Use the tool."""
        return self.func(*args, **kwargs)

    async def _arun(
            self,
            *args: Any,
            **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        if self.coroutine:
            return await self.coroutine(*args, **kwargs)
        raise NotImplementedError("Tool does not support async")

    @classmethod
    def from_function(
            cls,
            func: Callable,
            name: Optional[str] = None,
            description: Optional[str] = None,
            args_schema: Optional[Type[BaseModel]] = None,
            infer_schema: bool = True,
            **kwargs: Any,
    ) -> StructuredTool:
        name = name or func.__name__
        description = description or func.__doc__
        assert (
                description is not None
        ), "Function must have a docstring if description not provided."

        # Description example:
        # search_api(query: str) - Searches the API for the query.
        description = f"{name}{signature(func)} - {description.strip()}"
        _args_schema = args_schema
        if _args_schema is None and infer_schema:
            _args_schema = create_schema_from_function(f"{name}Schema", func)
        if iscoroutinefunction(func):
            return cls(
                name=name,
                func=func,
                coroutine=func,
                args_schema=_args_schema,
                description=description,
                **kwargs,
            )
        return cls(
            name=name,
            func=func,
            args_schema=_args_schema,
            description=description,
            **kwargs,
        )


def tool(
        *args: Union[str, Callable],
        args_schema: Optional[Type[BaseModel]] = None,
        infer_schema: bool = True,
        description: Optional[str] = None,
) -> Callable:
    """Make tools out of functions, can be used with or without arguments.

    Args:
        *args: The arguments to the tool.
        args_schema: optional argument schema for user to specify
        infer_schema: Whether to infer the schema of the arguments from
            the function's signature. This also makes the resultant tool
            accept a dictionary input to its `run()` function.

    Requires:
        - Function must be of type (str) -> str
        - Function must have a docstring

    Examples:
        .. code-block:: python

            @tool
            def search_api(query: str) -> str:
                # Searches the API for the query.
                return

            @tool("search")
            def search_api(query: str) -> str:
                # Searches the API for the query.
                return
    """

    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(func: Callable) -> BaseTool:
            if infer_schema or args_schema is not None:
                return StructuredTool.from_function(
                    func,
                    name=tool_name,
                    args_schema=args_schema,
                    infer_schema=infer_schema,
                    description=description,
                )
            # If someone doesn't want a schema applied, we must treat it as
            # a simple string->string function
            assert func.__doc__ is not None, "Function must have a docstring"
            return Tool(
                name=tool_name,
                func=func,
                description=f"{tool_name} tool" if description is None else description
            )

        return _make_tool

    if len(args) == 1 and isinstance(args[0], str):
        # if the argument is a string, then we use the string as the tool name
        # Example usage: @tool("search")
        return _make_with_name(args[0])
    elif len(args) == 1 and callable(args[0]):
        # if the argument is a function, then we use the function name as the tool name
        # Example usage: @tool
        return _make_with_name(args[0].__name__)(args[0])
    elif len(args) == 0:
        # if there are no arguments, then we use the function name as the tool name
        # Example usage: @tool()
        def _partial(func: Callable[[str], str]) -> BaseTool:
            return _make_with_name(func.__name__)(func)

        return _partial
    else:
        raise ValueError("Too many arguments for tool decorator")
