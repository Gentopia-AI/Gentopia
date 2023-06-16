import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = AutoTokenizer.from_pretrained(loader_model.base_url)

    if loader_model.device == "cpu":
        print("cpu mode")
        model = AutoModelForCausalLM.from_pretrained(
            loader_model.base_url,
            device_map={"": "cpu"},
            use_safetensors=False
        )

    elif loader_model.device == "mps":
        print("mps mode")
        model = AutoModelForCausalLM.from_pretrained(
            loader_model.base_url,
            device_map={"": "mps"},
            torch_dtype=torch.float16,
            use_safetensors=False
        )

    else:
        print("gpu mode")
        model = AutoModelForCausalLM.from_pretrained(
            loader_model.base_url,
            load_in_8bit=True if loader_model.device == "gpu-8bit" else False,
            load_in_4bit=True if loader_model.device == "gpu-4bit" else False,
            torch_dtype=torch.float16,
            device_map="auto",
            use_safetensors=False
        )

        if loader_model.device == "gpu":
            model.half()

    # model = BetterTransformer.transform(model)
    return model, tokenizer
