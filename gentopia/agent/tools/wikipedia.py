from typing import Any, Tuple

from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

from gentopia.agent.tools.tool import Tool


class Wikipedia(Tool):
    def __init__(self, name="Wikipedia", description=None, docstore=None):
        super().__init__(name, False)
        self.description = "Worker that search for similar page contents from Wikipedia. Useful when you need to " \
                           "get holistic knowledge about people, places, companies, historical events, " \
                           "or other subjects. The response are long and might contain some irrelevant information. " \
                           "Input should be a search query." if not description else description
        self.docstore = docstore

    def convert_input(self, args: Any) -> Tuple[bool, Any]:
        if isinstance(args, str) or '__str__' in dir(args):
            return True, str(args)
        return False, args

    def check_output(self, evidence) -> bool:
        return True

    def run(self, args, rectify=False):
        if not self.docstore:
            self.docstore = DocstoreExplorer(Wikipedia())
        result, args = self.convert_input(args)
        assert result, "Input must be a string"
        tool = self.docstore
        evidence = tool.search(args)
        self.cost += 1
        if not rectify and not self.check_output(evidence):
            args = self.rectify(args, evidence)
            evidence = self.run(args, True)
        return evidence

#
# if __name__ == '__main__':
#     worker = WikipediaWorker()
#     print(worker.run("Apple"))
