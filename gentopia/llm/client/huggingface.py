import json
import os
from typing import Generator

from pydantic import validator

from gentopia.llm.base_llm import BaseLLM
from gentopia.model.completion_model import *
from gentopia.model.param_model import *

# Load model data from resource
model_data = json.load(
    open(os.path.join(os.path.dirname(__file__), "../../resource/model_cards.json"), encoding="utf-8"))


class HuggingfaceLoader(BaseModel):
    """
    Huggingface loader
    """
    model_name: str
    device: str  # cpu, mps, gpu, gpu-8bit, gpu-4bit

    @validator('model_name')
    def validate_model_name(cls, v):
        if v not in model_data.keys():
            raise ValueError(f"model_name {v} is not supported")
        return v

    @validator('device')
    def validate_device(cls, v):
        if v not in ["cpu", "mps", "gpu", "gpu-8bit", "gpu-4bit"]:
            raise ValueError(f"device {v} is not supported")
        # Try to downgrade if no GPU is available
        if not torch.cuda.is_available() and v in ["gpu", "gpu-8bit", "gpu-4bit"]:
            print("GPU is not available. Switching to CPU mode")
            v = "cpu"
        return v

    def get_model_info(self) -> HuggingfaceLoaderModel:
        model_info = model_data[self.model_name]
        return HuggingfaceLoaderModel(model_name=self.model_name,
                                      description=model_info.get("desc", ""),
                                      base_url=model_info.get("hub(base)", ""),
                                      ckpt_url=model_info.get("hub(ckpt)", ""),
                                      device=self.device)

    def get_vram_usage(self):
        model_info = model_data[self.model_name]
        return {"half": model_info.get("vram(full)", ""),
                "8bit": model_info.get("vram(8bit)", ""),
                "4bit": model_info.get("vram(4bit)", "")}

    def load_model(self):
        """
        Map base url into mode loader.
        Output is a tuple of (model, tokenizer)
        """
        model_info = self.get_model_info()
        if "airoboros" in self.model_name:
            from gentopia.llm.loaders.airoboros import load_model
            return load_model(model_info)
        elif "alpaca" in self.model_name or "wizardlm" in self.model_name:
            from gentopia.llm.loaders.alpaca import load_model
            return load_model(model_info)
        elif "baize" in self.model_name:
            from gentopia.llm.loaders.baize import load_model
            return load_model(model_info)
        elif "bloom" in self.model_name:
            from gentopia.llm.loaders.bloom import load_model
            return load_model(model_info)
        elif "camel" in self.model_name:
            from gentopia.llm.loaders.camel import load_model
            return load_model(model_info)
        elif "falcon" in self.model_name:
            from gentopia.llm.loaders.falcon import load_model
            return load_model(model_info)
        elif "flan" in self.model_name:
            from gentopia.llm.loaders.flan_alpaca import load_model
            return load_model(model_info)
        elif "guanaco" in self.model_name:
            from gentopia.llm.loaders.guanaco import load_model
            return load_model(model_info)
        elif "kullm" in self.model_name:
            from gentopia.llm.loaders.kullm import load_model
            return load_model(model_info)
        elif "mpt" in self.model_name:
            from gentopia.llm.loaders.mpt import load_model
            return load_model(model_info)
        elif "redpajama" in self.model_name:
            from gentopia.llm.loaders.redpajama import load_model
            return load_model(model_info)
        elif "replit" in self.model_name:
            from gentopia.llm.loaders.replit import load_model
            return load_model(model_info)
        elif "samantha" in self.model_name:
            from gentopia.llm.loaders.samantha_vicuna import load_model
            return load_model(model_info)
        elif "stablelm" in self.model_name:
            from gentopia.llm.loaders.stablelm import load_model
            return load_model(model_info)
        elif "starchat" in self.model_name:
            from gentopia.llm.loaders.starchat import load_model
            return load_model(model_info)
        elif "t5-vicuna" in self.model_name:
            from gentopia.llm.loaders.t5_vicuna import load_model
            return load_model(model_info)
        elif "vicuna" in self.model_name:
            from gentopia.llm.loaders.samantha_vicuna import load_model
            return load_model(model_info)
        else:
            return None


class HuggingfaceLLMClient(BaseLLM, BaseModel):
    """
    Huggingface LLM client
    """
    model_name: str
    params: HuggingfaceParamModel
    device: str  # cpu, mps, gpu, gpu-8bit, gpu-4bit

    @validator('device')
    def validate_device(cls, v):
        if v not in ["cpu", "mps", "gpu", "gpu-8bit", "gpu-4bit"]:
            raise ValueError(f"device {v} is not supported")
        # Try to downgrade if no GPU is available
        if not torch.cuda.is_available() and v in ["gpu", "gpu-8bit", "gpu-4bit"]:
            print("GPU is not available. Switching to CPU mode")
            v = "cpu"
        return v

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_param(self) -> BaseParamModel:
        return self.params

    def get_model_loader_info(self) -> HuggingfaceLoaderModel:
        model_loader = HuggingfaceLoader(model_name=self.model_name, device=self.device)
        return model_loader.get_model_info()

    def completion(self, prompt: str, **kwargs) -> BaseCompletion:
        """
        Generate completion
        """
        # Load model
        model_loader = HuggingfaceLoader(model_name=self.model_name, device=self.device)
        loads = model_loader.load_model()
        if loads is None:
            raise ValueError(f"model {self.model_name} is not supported")
        model, tokenizer = loads
        # Generate completion
        if self.device in ["gpu", "gpu-8bit", "gpu-4bit"]:
            inputs = tokenizer(prompt, return_tensors="pt").to(torch.device("cuda"))
        else:
            inputs = tokenizer(prompt, return_tensors="pt")

        try:
            outputs = model.generate(inputs=inputs.input_ids,
                                     temperature=self.params.temperature,
                                     top_p=self.params.top_p,
                                     max_new_tokens=self.params.max_new_tokens,
                                     **kwargs
                                     )
            completion = tokenizer.decode(outputs[:, inputs.input_ids.shape[-1]:][0], skip_special_tokens=True)
            n_input_tokens = inputs.input_ids.shape[1]
            n_output_tokens = outputs.shape[1]
            return BaseCompletion(state="success",
                                  content=completion,
                                  prompt_token=n_input_tokens,
                                  completion_token=n_output_tokens)
        except Exception as e:
            return BaseCompletion(state="error", content=str(e))

    def chat_completion(self, message) -> ChatCompletion:
        # TODO: Implement chat_completion
        raise NotImplementedError("chat_completion is not supported for Huggingface LLM")

    def stream_chat_completion(self, message) -> Generator:
        # TODO: Implement stream_chat_completion
        raise NotImplementedError("stream_chat_completion is not supported for Huggingface LLM")
