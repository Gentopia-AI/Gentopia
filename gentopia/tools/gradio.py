from typing import AnyStr
from langchain import OpenAI, LLMMathChain
from .basetool import *
from xml.dom.pulldom import SAX2DOM
from gradio_tools.tools import BarkTextToSpeechTool,StableDiffusionTool,DocQueryDocumentAnsweringTool,ImageCaptioningTool,StableDiffusionPromptGeneratorTool,TextToVideoTool,ImageToMusicTool,WhisperAudioTranscriptionTool,ClipInterrogatorTool
from gradio_client.client import Job
from gradio_client.utils import QueueError
import time


class TTS(BaseTool):
    name = "text-to-speech"
    description = "Converting text into sounds that sound like a human read it"
    args_schema: Optional[Type[BaseModel]] = create_model("TTSArgs", text=(str, ...))
    
    def _run(self, text: AnyStr) -> Any:
        bk = BarkTextToSpeechTool()
        return bk.run(text)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class VisualQA(BaseTool):
    name = "VisualQA"
    description = "Answer a question from the given image"
    args_schema: Optional[Type[BaseModel]] = create_model("VisualQAArgs", path_to_image=(str, ...), question=(str, ...))

    def _run(self, path_to_image: AnyStr, question: AnyStr) -> Any:
        ans = DocQueryDocumentAnsweringTool().run(f"{path_to_image},{question}")
        return ans

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class ImageCaption(BaseTool):
    name = "image_caption"
    description = "Generating a caption for an image"
    args_schema: Optional[Type[BaseModel]] = create_model("ImageCaptioningArgs", path_to_image=(str, ...))

    def _run(self, path_to_image: AnyStr) -> Any:
        ans = ImageCaptioningTool().run(f"{path_to_image}")
        return ans

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class TextToImage(BaseTool):
    name = "Text2Image"
    description = "generate images based on text input"
    args_schema: Optional[Type[BaseModel]] = create_model("TextToImageArgs", text=(str, ...))

    def _run(self, text: AnyStr) -> Any:
        ans = StableDiffusionTool().run(text)
        return ans

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class TextToVideo(BaseTool):
    name = "Text2Video"
    description = "generate videos based on text input"
    args_schema: Optional[Type[BaseModel]] = create_model("Text2VideoArgs", text=(str, ...))

    def _run(self, text: AnyStr) -> Any:
        ans = TextToVideoTool().run(text)
        return ans

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class AudioToText(BaseTool):
    name = "Audio2Text"
    description = "transcribing an audio file into text transcript"
    args_schema: Optional[Type[BaseModel]] = create_model("Audio2TextArgs", path_to_audio=(str, ...))

    def _run(self, path_to_audio: AnyStr) -> Any:
        ans = WhisperAudioTranscriptionTool().run(path_to_audio)
        return ans

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class ImageToPrompt(BaseTool):
    name = "Image2Prompt"
    description = "creating a prompt for StableDiffusion that matches the input image"
    args_schema: Optional[Type[BaseModel]] = create_model("Image2PromptArgs", path_to_image=(str, ...))

    def _run(self, path_to_image: AnyStr) -> Any:
        ans = ClipInterrogatorTool().run(path_to_image)
        return ans

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    

# @tool.get("/get_audiotrans")
# def imgtomsc(input : str)-> str:
#     '''Transcribing an audio file track into text transcript.
#     '''
#     at = WhisperAudioTranscriptionTool()
#     return at.run(input)
# @tool.get("/get_imgprompt")
# def imgprompt(input : str)-> str:
#     '''Creating a prompt for StableDiffusion that matches the input image.
#     '''
#     ci = ClipInterrogatorTool()
#     return ci.run(input)
# return tool


if __name__ == "__main__":
    ans = TTS()._run("Please surprise me and speak in whatever voice you enjoy. Vielen Dank und Gesundheit!")
    # ans = VisualQA()._run("tools/image.jpg", "what does the image contain ?")
    # ans = ImageCaption()._run("tools/image.jpg")
    ans = TextToImage()._run("an asian student wearing a black t-shirt")
    # ans = TextToVideo()._run("an asian student wearing a black t-shirt")
    # ans = AudioToText()._run("/var/folders/xv/d1zbl5b50p58ttlb7bh655340000gn/T/tmpx0senrsmo38ocexe.wav")
    # ans = ImageToPrompt()._run("image.jpg")
    print(ans)
