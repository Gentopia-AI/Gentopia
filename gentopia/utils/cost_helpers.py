from gentopia.llm.llm_info import *


def calculate_cost(model_name: str, prompt_token: int, completion_token: int) -> float:
    """
    Calculate the cost of a prompt and completion.

    Args:
        model_name: The name of the model.
        prompt_token: The number of prompt tokens.
        completion_token: The number of completion tokens.

    Returns:
        The calculated cost.
    """
    # 0 if model_name is not in COSTS
    return COSTS.get(model_name, dict()).get("prompt", 0) * prompt_token \
        + COSTS.get(model_name, dict()).get("completion", 0) * completion_token
