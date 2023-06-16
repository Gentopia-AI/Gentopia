import torch
from optimum.bettertransformer import BetterTransformer
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = LlamaTokenizer.from_pretrained(loader_model.base_url)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if loader_model.device == "cpu":
        print("cpu mode")
        model = LlamaForCausalLM.from_pretrained(
            loader_model.base_url,
            device_map={"": "cpu"},
            use_safetensors=False
        )

        if loader_model.ckpt_url:
            model = PeftModel.from_pretrained(
                model,
                loader_model.ckpt_url,
                device_map={"": "cpu"}
            )
        else:
            model = BetterTransformer.transform(model)

    elif loader_model.device == "mps":
        print("mps mode")
        model = LlamaForCausalLM.from_pretrained(
            loader_model.base_url,
            device_map={"": "mps"},
            torch_dtype=torch.float16,
            use_safetensors=False
        )

        if loader_model.ckpt_url:
            model = PeftModel.from_pretrained(
                model,
                loader_model.ckpt_url,
                torch_dtype=torch.float16,
                device_map={"": "mps"}
            )
        else:
            model = BetterTransformer.transform(model)

    else:
        print("gpu mode")
        model = LlamaForCausalLM.from_pretrained(
            loader_model.base_url,
            load_in_8bit=True if loader_model.device == "gpu-8bit" else False,
            load_in_4bit=True if loader_model.device == "gpu-4bit" else False,
            torch_dtype=torch.float16,
            device_map="auto",
            use_safetensors=False
        )

        if loader_model.device == "gpu":
            model.half()

        if loader_model.ckpt_url:
            model = PeftModel.from_pretrained(
                model,
                loader_model.ckpt_url,
                # force_download=force_download_ckpt,
            )
        else:
            model = BetterTransformer.transform(model)

    return model, tokenizer
