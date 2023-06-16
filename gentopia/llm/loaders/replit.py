import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from gentopia.model.param_model import HuggingfaceLoaderModel


def load_model(loader_model: HuggingfaceLoaderModel):
    tokenizer = AutoTokenizer.from_pretrained(loader_model.base_url, trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        loader_model.base_url,
        load_in_8bit=True if loader_model.device == "gpu-8bit" else False,
        load_in_4bit=True if loader_model.device == "gpu-4bit" else False,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if loader_model.ckpt_url:
        model = PeftModel.from_pretrained(
            model,
            loader_model.ckpt_url,
            # force_download=force_download_ckpt,
            trust_remote_code=True
        )

        model = model.merge_and_unload()

    # model = BetterTransformer.transform(model)
    model.to('cuda')
    return model, tokenizer
