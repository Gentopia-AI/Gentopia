from gentopia.model.agent_model import AgentOutput


def regularize_block(block):
    """
    Strip and add \n to the end of the block.
    """
    return block.strip("\n") + "\n"


def get_plugin_response_content(output) -> str:
    if isinstance(output, AgentOutput):
        return output.output
    else:
        return str(output)
