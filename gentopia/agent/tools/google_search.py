from typing import Union, AnyStr

from langchain import SerpAPIWrapper

from gentopia.agent.tools.tool import Tool


class GoogleSearch(Tool):
    def __init__(self, name="GoogleSearch", description: Union[AnyStr, None] = None):
        super().__init__(name, False)
        self.description = "Worker that search for similar page contents from Wikipedia. Useful when you need to " \
                           "get holistic knowledge about people, places, companies, historical events, " \
                           "or other subjects. The response are long and might contain some irrelevant information. " \
                           "Input should be a search query." if not description else description

    def convert_input(self, args):
        if isinstance(args, str) or '__str__' in dir(args):
            return True, str(args)
        return False, args

    def check_output(self, evidence) -> bool:
        return True

    def run(self, args, rectify=False):
        result, args = self.convert_input(args)
        assert result, "Input must be a string"
        tool = SerpAPIWrapper()
        evidence = tool.run(args)
        self.cost += 1
        if not rectify and not self.check_output(evidence):
            args = self.rectify(args, evidence)
            evidence = self.run(args, True)
        return evidence


if __name__ == '__main__':
    worker = GoogleSearch()
    print(worker.run("Barack Obama"))
