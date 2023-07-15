from gentopia.model.agent_model import AgentOutput


def regularize_block(block):
    """
    Regularize a block by stripping and adding a newline character at the end.

    Args:
        block: The block of text to be regularized.

    Returns:
        The regularized block with a newline character at the end.
    """
    return block.strip("\n") + "\n"


def get_plugin_response_content(output) -> str:
    """
    Get the content of a plugin response.

    Args:
        output: The output from a plugin.

    Returns:
        The content of the plugin response as a string.
    """
    if isinstance(output, AgentOutput):
        return output.output
    else:
        return str(output)
