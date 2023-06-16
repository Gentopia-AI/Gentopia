import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = AutoTokenizer.from_pretrained(loader_model.base_url)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if loader_model.device == "cpu":
        print("cpu mode")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            loader_model.base_url,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True
        )

    elif loader_model.device == "mps":
        print("mps mode")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            loader_model.base_url,
            device_map={"": "mps"},
            torch_dtype=torch.float16,
        )

    else:
        print("gpu mode")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            loader_model.base_url,
            load_in_8bit=True if loader_model.device == "gpu-8bit" else False,
            load_in_4bit=True if loader_model.device == "gpu-4bit" else False,
            device_map="auto",
        )

        if loader_model.device == "gpu":
            model.half()

    model = BetterTransformer.transform(model)
    return model, tokenizer
