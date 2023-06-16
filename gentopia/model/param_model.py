from typing import Union, Dict

import torch
from pydantic import BaseModel


class BaseParamModel(BaseModel):
    model_name: str


class OpenAIParamModel(BaseModel):
    """
    OpenAI API parameters
    """
    model_name: str = "gpt-4"
    max_tokens: int = 4032
    temperature: float = 0.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    n: int = 1
    stop: list = []


class HuggingfaceLoaderModel(BaseModel):
    """
    Huggingface model loading parameters.
    """
    model_name: str
    description: str
    base_url: str
    ckpt_url: str
    device: str  # "cpu", "mps", "gpu", "gpu-8bit", "gpu-4bit"

    @property
    def device_map(self) -> Union[Dict[str, str], str]:
        device = {"", self.device}
        if self.device.startswith("gpu"):
            device = "auto"
        return device

    @property
    def default_args(self):
        args, kwargs = [self.base_url], dict(
            device_map={"": self.device},
            use_safetensors=False,
        )
        if self.device != "cpu":
            kwargs['torch_dtype'] = torch.float16
        if self.device.startswith("gpu"):
            kwargs['load_in_8bit'] = True if self.device == "gpu-8bit" else False
            kwargs['load_in_4bit'] = True if self.device == "gpu-4bit" else False
            kwargs['device_map'] = "auto"
        return args, kwargs


class HuggingfaceParamModel(BaseParamModel):
    """
    basic Huggingface inference parameters.
    """
    model_name: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
