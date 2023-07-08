from __future__ import annotations

from gradio_client.client import Job

from gradio_tools.tools.gradio_tool import GradioTool


class StableDiffusionPromptGeneratorTool(GradioTool):
    def __init__(
        self,
        name="StableDiffusionPromptGenerator",
        description=(
            "Use this tool to improve a prompts for stable diffusion and other image and video generators. "
            "This tool will refine your prompts to include key words and phrases that make "
            "stable diffusion and other art generation algorithms perform better. The input is a prompts text string "
            "and the output is a prompts text string"
        ),
        src="microsoft/Promptist",
        hf_token=None,
        duplicate=False,
    ) -> None:
        super().__init__(name, description, src, hf_token, duplicate)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query, api_name="/predict")

    def postprocess(self, output: str) -> str:
        return output
