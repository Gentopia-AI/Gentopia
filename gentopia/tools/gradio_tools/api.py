from typing import AnyStr

from gentopia.tools.basetool import *
from gentopia.tools.gradio_tools.tools import BarkTextToSpeechTool


class TTS(BaseTool):
    name = "text-to-speech"
    description = "Converting text into sounds that sound like a human read it"
    args_schema: Optional[Type[BaseModel]] = create_model("TTSArgs", text=(str, ...))
    
    def _run(self, text: AnyStr) -> Any:
        bk = BarkTextToSpeechTool()
        return bk.run(text)


# @tool.get("/get_qa")
# def qa(input : str)-> str:
#     '''Answering questions from the image of the document.
#     '''
#     qa = DocQueryDocumentAnsweringTool()
#     return qa.run(input)
# @tool.get("/get_imagecaption")
# def imagecaption(input : str)-> str:
#     '''Creating a caption for an image.
#     '''
#     ic = ImageCaptioningTool()
#     return ic.run(input)
# @tool.get("/get_promptgenerator")
# def promptgenerator(input : str)-> str:
#     '''Generating a prompt for stable diffusion and other image and video generators based on text input.
#     '''
#     pg = StableDiffusionPromptGeneratorTool()
#     return pg.run(input)
# @tool.get("/get_stablediffusion")
# def stablediffusion(input : str)-> str:
#     '''generate images based on text input.
#     '''
#     sd = StableDiffusionTool()
#     return sd.run(input)
# @tool.get("/get_texttovideo")
# def texttovideo(input : str)-> str:
#     '''Creating videos from text.
#     '''
#     tv = TextToVideoTool()
#     return tv.run(input)
# @tool.get("/get_imgtomsc")
# def imgtomsc(input : str)-> str:
#     '''Creating music from images.
#     '''
#     im = ImageToMusicTool()
#     return im.run(input)
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
    ans = TTS()._run("hello")
    print(ans)
