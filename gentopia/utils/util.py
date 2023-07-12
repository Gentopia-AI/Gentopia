from gentopia.model.param_model import BaseParamModel
from gentopia.prompt import fewshots

# OPENAI_COMPLETION_MODELS = ["text-davinci-003"]
# OPENAI_CHAT_MODELS = ["gpt-3.5-turbo", "gpt-4"]
# LLAMA_WEIGHTS = ["tloen/alpaca-lora-7b", "rewoo/planner_7B"]
#
# DEFAULT_EXEMPLARS_COT = {"hotpot_qa": fewshots.HOTPOTQA_COT,
#                          "trivia_qa": fewshots.TRIVIAQA_COT,
#                          "gsm8k": fewshots.GSM8K_COT,
#                          "physics_question": fewshots.TRIVIAQA_COT,
#                          "sports_understanding": fewshots.TRIVIAQA_COT,
#                          "strategy_qa": fewshots.TRIVIAQA_COT,
#                          "sotu_qa": fewshots.TRIVIAQA_COT}
#
# DEFAULT_EXEMPLARS_REACT = {"hotpot_qa": fewshots.HOTPOTQA_REACT,
#                            "trivia_qa": fewshots.TRIVIAQA_REACT,
#                            "gsm8k": fewshots.GSM8K_REACT,
#                            "physics_question": fewshots.GSM8K_REACT,
#                            "sports_understanding": fewshots.GSM8K_REACT,
#                            "strategy_qa": fewshots.GSM8K_REACT,
#                            "sotu_qa": fewshots.GSM8K_REACT}
#
# DEFAUL_EXEMPLARS_PWS = {"hotpot_qa": fewshots.HOTPOTQA_PWS_BASE,
#                         "trivia_qa": fewshots.TRIVIAQA_PWS,
#                         "gsm8k": fewshots.GSM8K_PWS,
#                         "physics_question": fewshots.GSM8K_PWS,
#                         "sports_understanding": fewshots.GSM8K_PWS,
#                         "strategy_qa": fewshots.GSM8K_PWS,
#                         "sotu_qa": fewshots.GSM8K_PWS}
#
#
# def get_token_unit_price(model):
#     if model in OPENAI_COMPLETION_MODELS:
#         return 0.00002
#     elif model in OPENAI_CHAT_MODELS:
#         if model == "gpt-3.5-turbo":
#             return 0.000002
#         elif model == "gpt-4":
#             return 0.00003
#     elif model in LLAMA_WEIGHTS:
#         return 0.0
#     else:
#         raise ValueError("Model not found")


#TODO: get default client param model
def get_default_client_param_model(model_name:str) -> BaseParamModel:
    return None

def print_tree(obj, indent=0):
    for attr in dir(obj):
        if not attr.startswith('_'):
            value = getattr(obj, attr)
            if not callable(value):
                if not isinstance(value, dict) and not isinstance(value, list):
                    print('|   ' * indent + '|--', f'{attr}: {value}')
                else:
                    if not value:
                        print('|   ' * indent + '|--', f'{attr}: {value}')
                    print('|   ' * indent + '|--', f'{attr}:')
                if hasattr(value, '__dict__'):
                    print_tree(value, indent + 1)
                elif isinstance(value, list):
                    for item in value:
                        print_tree(item, indent + 1)
                elif isinstance(value, dict):
                    for key, item in value.items():
                        print_tree(item, indent + 1)