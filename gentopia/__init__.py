import signal
from .prompt import PromptTemplate
from .assembler.agent_assembler import AgentAssembler
from .output import enable_log
from .output.base_output import *
from .output.console_output import ConsoleOutput


def chat(agent, output = ConsoleOutput(), verbose = False, log_level = None, log_path = None):
    output.panel_print("[green]Welcome to Gentopia!", title="[blue]Gentopia")
    if verbose:
        output.panel_print(str(agent), title="[red]Agent")
    if log_level is not None:
        if not check_log():
            enable_log( path=log_path, log_level=log_level)
    def handler(signum, frame):
        output.print("\n[red]Bye!")
        exit(0)

    signal.signal(signal.SIGINT, handler)

    while True:
        output.print("[green]User: ", end="")
        text = input()
        if text:
            response = agent.stream(text, output=output)
        else:
            response = agent.stream(output=output)

        output.done(_all=True)

