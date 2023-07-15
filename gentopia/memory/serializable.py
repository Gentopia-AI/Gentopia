from pydantic import BaseModel, PrivateAttr
from abc import ABC
from typing import Any, Dict, List, Literal, TypedDict, Union, cast


class BaseSerialized(TypedDict):
    gt: int
    id: List[str]


class SerializedConstructor(BaseSerialized):
    type: Literal["constructor"]
    kwargs: Dict[str, Any]


class SerializedSecret(BaseSerialized):
    type: Literal["secret"]


class SerializedNotImplemented(BaseSerialized):
    type: Literal["not_implemented"]


class Serializable(BaseModel, ABC):
    @property
    def gt_serializable(self) -> bool:
        """
        Return whether or not the class is serializable.
        """
        return False

    @property
    def gt_namespace(self) -> List[str]:
        """
        Return the namespace of the gentopia object.
        """
        return self.__class__.__module__.split(".")

    @property
    def gt_secrets(self) -> Dict[str, str]:
        """
        Return a map of constructor argument names to secret ids.
        eg. {"openai_api_key": "OPENAI_API_KEY"}
        """
        return dict()

    @property
    def gt_attributes(self) -> Dict:
        """
        Return a list of attribute names that should be included in the
        serialized kwargs. These attributes must be accepted by the
        constructor.
        """
        return {}

    class Config:
        extra = "ignore"

    _gt_kwargs = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Serializable object.

        Args:
            **kwargs (Any): Keyword arguments to initialize the object.
        """
        super().__init__(**kwargs)
        self._gt_kwargs = kwargs

    def to_json(self) -> Union[SerializedConstructor, SerializedNotImplemented]:
        """
        Convert the object to JSON representation.

        Returns:
            Union[SerializedConstructor, SerializedNotImplemented]: The JSON representation.

        Notes:
            - If the object is not serializable, returns SerializedNotImplemented.
            - If the object is serializable, returns SerializedConstructor.
        """
        if not self.gt_serializable:
            return self.to_json_not_implemented()

        secrets = dict()
        # Get latest values for kwargs if there is an attribute with same name
        gt_kwargs = {
            k: getattr(self, k, v)
            for k, v in self._gt_kwargs.items()
            if not (self.__exclude_fields__ or {}).get(k, False)  # type: ignore
        }

        # Merge the gt_secrets and gt_attributes from every class in the MRO
        for cls in [None, *self.__class__.mro()]:
            # Once we get to Serializable, we're done
            if cls is Serializable:
                break

            # Get a reference to self bound to each class in the MRO
            this = cast(
                Serializable, self if cls is None else super(cls, self))

            secrets.update(this.gt_secrets)
            gt_kwargs.update(this.gt_attributes)

        # include all secrets, even if not specified in kwargs
        # as these secrets may be passed as an environment variable instead
        for key in secrets.keys():
            secret_value = getattr(self, key, None) or gt_kwargs.get(key)
            if secret_value is not None:
                gt_kwargs.update({key: secret_value})

        return {
            "gt": 1,
            "type": "constructor",
            "id": [*self.gt_namespace, self.__class__.__name__],
            "kwargs": gt_kwargs
            if not secrets
            else _replace_secrets(gt_kwargs, secrets),
        }

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        """
        Convert the object to a not implemented JSON representation.

        Returns:
            SerializedNotImplemented: The not implemented JSON representation.
        """
        return to_json_not_implemented(self)


def _replace_secrets(
    root: Dict[Any, Any], secrets_map: Dict[str, str]
) -> Dict[Any, Any]:
    """
    Replace secrets in the JSON representation.

    Args:
        root (Dict[Any, Any]): The root dictionary.
        secrets_map (Dict[str, str]): The map of secrets.

    Returns:
        Dict[Any, Any]: The dictionary with replaced secrets.
    """
    result = root.copy()
    for path, secret_id in secrets_map.items():
        [*parts, last] = path.split(".")
        current = result
        for part in parts:
            if part not in current:
                break
            current[part] = current[part].copy()
            current = current[part]
        if last in current:
            current[last] = {
                "gt": 1,
                "type": "secret",
                "id": [secret_id],
            }
    return result


def to_json_not_implemented(obj: object) -> SerializedNotImplemented:
    """
    Convert an object to a not implemented JSON representation.

    Args:
        obj (object): The object to convert.

    Returns:
        SerializedNotImplemented: The not implemented JSON representation.
    """
    _id: List[str] = []
    try:
        if hasattr(obj, "__name__"):
            _id = [*obj.__module__.split("."), obj.__name__]
        elif hasattr(obj, "__class__"):
            _id = [
                *obj.__class__.__module__.split("."), obj.__class__.__name__]
    except Exception:
        pass
    return {
        "gt": 1,
        "type": "not_implemented",
        "id": _id,
    }
