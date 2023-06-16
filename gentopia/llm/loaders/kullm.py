from transformers import AutoModelForCausalLM, AutoTokenizer

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = AutoTokenizer.from_pretrained(loader_model.base_url)
    args, kwargs = loader_model.default_args
    model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
    if loader_model.device == "gpu":
        model.half()
    return model, tokenizer
