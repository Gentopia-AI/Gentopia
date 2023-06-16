from transformers import AutoModelForCausalLM, AutoTokenizer

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = AutoTokenizer.from_pretrained(loader_model.base_url, trust_remote_code=True)
    tokenizer.padding_side = "left"
    args, kwargs = loader_model.default_args
    kwargs['trust_remote_code'] = True
    model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
    if loader_model.device == "gpu":
        model.half()
    return model, tokenizer
