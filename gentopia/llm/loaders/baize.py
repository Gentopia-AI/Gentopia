import torch
from optimum.bettertransformer import BetterTransformer
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = LlamaTokenizer.from_pretrained(loader_model.base_url)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    args, kwargs = loader_model.default_args
    model = LlamaForCausalLM.from_pretrained(*args, **kwargs)
    if loader_model.device == "gpu":
        model.half()
    if loader_model.ckpt_url:
        args, kwargs = [model, loader_model.ckpt_url], dict(device_map=loader_model.device_map)
        if loader_model.device == "mps":
            kwargs.update(torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(*args, **kwargs)
    else:
        model = BetterTransformer.transform(model)
    return model, tokenizer
