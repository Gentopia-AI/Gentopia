from typing import AnyStr, Any, Union, Tuple


class Tool:
    def __init__(self, name: AnyStr, isLLMBased: bool = False,
                 description: Union[AnyStr, None] = None):
        self.name = name
        self.cost = 0
        self.isLLMBased = isLLMBased
        self.description = description
        self.format = dict(name="tool", args=dict())

    def run(self, args: Any, rectify: bool = False) -> Any:
        raise NotImplementedError

    def convert_input(self, args: Any) -> Tuple[bool, Any]:
        raise NotImplementedError

    def check_output(self, evidence) -> bool:
        raise NotImplementedError

    def rectify(self, args, evidence):
        pass

    def from_dict(self): pass
