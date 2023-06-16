import global_vars
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = AutoTokenizer.from_pretrained(
        loader_model.base_url, use_fast=False if global_vars.model_type == "stable-vicuna" else True
    )
    tokenizer.padding_side = "left"
    args, kwargs = loader_model.default_args
    model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
    if loader_model.device == "gpu":
        model.half()
    model = BetterTransformer.transform(model)
    return model, tokenizer
