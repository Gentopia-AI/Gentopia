from optimum.bettertransformer import BetterTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = AutoTokenizer.from_pretrained(loader_model.base_url)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    args, kwargs = loader_model.default_args
    kwargs.pop('use_safetensors')
    if loader_model.device == "cpu":
        kwargs['low_cpu_mem_usage'] = True
    model = AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
    if loader_model.device == "gpu":
        model.half()
    model = BetterTransformer.transform(model)
    return model, tokenizer
