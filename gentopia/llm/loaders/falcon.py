import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = AutoTokenizer.from_pretrained(loader_model.base_url)
    tokenizer.padding_side = "left"
    args, kwargs = loader_model.default_args
    kwargs['torch_dtype'] = torch.bfloat16
    kwargs['trust_remote_code'] = True
    model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
    return model, tokenizer
