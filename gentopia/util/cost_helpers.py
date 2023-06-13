from gentopia.llm.llm_info import *


def calculate_cost(model_name: str, prompt_token: int, completion_token: int) -> float:
    """
    Calculate cost of a prompt and completion.
    """
    return COSTS[model_name]["prompt"] * prompt_token + COSTS[model_name]["completion"] * completion_token
