import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = AutoTokenizer.from_pretrained(loader_model.base_url)
    tokenizer.padding_side = "left"

    if loader_model.device == "cpu":
        print("cpu mode")
        model = AutoModelForCausalLM.from_pretrained(
            loader_model.base_url,
            device_map={"": "cpu"},
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
            trust_remote_code=True
        )

    elif loader_model.device == "mps":
        print("mps mode")
        model = AutoModelForCausalLM.from_pretrained(
            loader_model.base_url,
            device_map={"": "mps"},
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
            trust_remote_code=True
        )

    else:
        print("gpu mode")
        model = AutoModelForCausalLM.from_pretrained(
            loader_model.base_url,
            load_in_8bit=True if loader_model.device == "gpu-8bit" else False,
            load_in_4bit=True if loader_model.device == "gpu-4bit" else False,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=False
        )

        # if not mode_8bit and not mode_4bit:
        #     model.half()

    # model = BetterTransformer.transform(model)
    return model, tokenizer
