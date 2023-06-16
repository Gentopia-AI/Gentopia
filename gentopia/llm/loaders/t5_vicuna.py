from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = T5Tokenizer.from_pretrained(loader_model.base_url, use_fast=False)
    tokenizer.padding_side = "left"
    args, kwargs = loader_model.default_args
    model = AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
    if loader_model.device == "gpu":
        model.half()
    model = BetterTransformer.transform(model)
    return model, tokenizer
