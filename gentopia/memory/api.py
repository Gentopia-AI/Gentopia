from gentopia.memory.vectorstores.vectorstore import VectorStoreRetrieverMemory
from gentopia.memory.base_memory import BaseMemory
from gentopia.memory.vectorstores.pinecone import Pinecone
from gentopia.memory.vectorstores.chroma import Chroma
from gentopia.memory.embeddings import OpenAIEmbeddings
from gentopia.llm.base_llm import BaseLLM
from gentopia import PromptTemplate
from gentopia.output.base_output import BaseOutput
import pydantic
import os
import queue

class Config:
    arbitrary_types_allowed = True

SummaryPrompt = PromptTemplate(
    input_variables=["rank", "input", "output"],
    template=
"""
You are a helpful assistant who is expected to summarize some sentences.
Another AI assistant is interacting with the user and multiple tools. 
Here is part of their conversation. You need to summarize it and provide a brief summary, which will help other assistants to recall their thoughts and actions.
Note that you need to use phrases like "In the fourth step" according to the rank to start your summary. For example, use "In the fifth step" if it is step 5, or use "First" if it is step 1, or "Second" in step 2.

In step {rank}, the part of the conversation is:
Input: {input}
Output: {output}
Your summary:
"""
)

FormerContextPrompt = PromptTemplate(
    input_variables=['summary'],
    template=
"""
The following summaries of context may help you recall the former conversation.
{summary}.
End of the summaries.
"""
)

RecallPrompt = PromptTemplate(
    input_variables=["summary"],
    template=
"""
The following summaries of context may help you recall your prior steps, which assist you in taking your next step.
{summary}
End of the summaries.
"""
)

RelatedContextPrompt = PromptTemplate(
    input_variables=["related_history"],
    template=
"""
Here are some related conversations that may help you answer the next question:
{related_history}
End of the related history.
"""
)

@pydantic.dataclasses.dataclass(config=Config)
class MemoryWrapper:
    """
    Wrapper class for memory management.
    """

    memory: BaseMemory
    conversation_threshold: int
    reasoning_threshold: int
    
    def __init__(self, memory: VectorStoreRetrieverMemory, conversation_threshold: int, reasoning_threshold: int):
        """
        Initialize the MemoryWrapper.

        Args:
            memory: The vector store retriever memory.
            conversation_threshold: The conversation threshold.
            reasoning_threshold: The reasoning threshold.
        """
        self.memory = memory
        self.conversation_threshold = conversation_threshold
        self.reasoning_threshold = reasoning_threshold
        assert self.conversation_threshold >= 0
        assert self.reasoning_threshold >= 0
        self.history_queue_I = queue.Queue()
        self.history_queue_II = queue.Queue()
        self.summary_I = ""     # memory I - level
        self.summary_II = ""    # memory II - level
        self.rank_I = 0
        self.rank_II = 0
    
    def __save_to_memory(self, io_obj):
        """
        Save the input-output pair to memory.

        Args:
            io_obj: The input-output pair.
        """
        self.memory.save_context(io_obj[0], io_obj[1]) # (input, output)
    
    def save_memory_I(self, query, response, output: BaseOutput):
        """
        Save the conversation to memory (level I).

        Args:
            query: The query.
            response: The response.
            output: The output object.
        """
        output.update_status("Conversation Memorizing...")
        self.rank_I += 1
        self.history_queue_I.put((query, response, self.rank_I))
        while self.history_queue_I.qsize() > self.conversation_threshold:
            top_context = self.history_queue_I.get()
            self.__save_to_memory(top_context)
            # self.summary_I += llm.completion(prompt=SummaryPrompt.format(rank=top_context[2], input=top_context[0], output=top_context[1])).content + "\n"
        output.done()

    def save_memory_II(self, query, response, output: BaseOutput, llm: BaseLLM):
        """
        Save the conversation to memory (level II).

        Args:
            query: The query.
            response: The response.
            output: The output object.
            llm: The BaseLLM object.
        """
        output.update_status("Reasoning Memorizing...")
        self.rank_II += 1
        self.history_queue_II.put((query, response, self.rank_II))
        while self.history_queue_II.qsize() > self.reasoning_threshold:
            top_context = self.history_queue_II.get()
            self.__save_to_memory(top_context)
            output.done()
            output.update_status("Summarizing...")
            self.summary_II += llm.completion(prompt=SummaryPrompt.format(rank=top_context[2], input=top_context[0], output=top_context[1])).content + "\n"
        output.done()

    def lastest_context(self, instruction, output: BaseOutput):
        """
        Get the latest context history.

        Args:
            instruction: The instruction.
            output: The output object.

        Returns:
            The context history.
        """
        context_history = []
        # TODO this context_history can only be used in openai agent. This function should be more universal
        if self.summary_I != "":
            context_history.append({"role": "system", "content": FormerContextPrompt.format(summary = self.summary_I)})
        for i in list(self.history_queue_I.queue):
            context_history.append(i[0])
            context_history.append(i[1])
        related_history = self.load_history(instruction)

        if related_history != "":
            output.panel_print(related_history, f"[green] Auto Conversation Memory: ")
            context_history.append({"role": "user", "content": RelatedContextPrompt.format(related_history=related_history)})

        context_history.append({"role": "user", "content": instruction})

        if self.summary_II != "":
            output.panel_print(self.summary_II, f"[green] Summary of Prior Steps: ")
            context_history.append({"role": "user", "content": RecallPrompt.format(summary = self.summary_II)})
            
        for i in list(self.history_queue_II.queue):
            context_history.append(i[0])
            context_history.append(i[1])
        return context_history

    def clear_memory_II(self):
        """
        Clear memory (level II).
        """
        self.summary_II = ""
        self.history_queue_II = queue.Queue()
        self.rank_II = 0
 
    
    def load_history(self, input):
        """
        Load history from memory.

        Args:
            input: The input.

        Returns:
            The loaded history.
        """
        return self.memory.load_memory_variables({"query": input})['history']


def create_memory(memory_type, conversation_threshold, reasoning_threshold, **kwargs) -> MemoryWrapper:
    """
    Create a memory object.

    Args:
        memory_type: The type of memory.
        conversation_threshold: The conversation threshold.
        reasoning_threshold: The reasoning threshold.
        **kwargs: Additional keyword arguments.

    Returns:
        The created MemoryWrapper object.
    """
    # choose desirable memory you need!
    memory: BaseMemory = None
    if memory_type == "pinecone":
        # according to params, initialize your memory.
        import pinecone
        embedding_fn = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]).embed_query
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"],environment=os.environ["PINECONE_ENVIRONMENT"])
        index = pinecone.Index(kwargs["index"])
        vectorstore = Pinecone(index, embedding_fn, kwargs["text_key"], namespace=kwargs.get("namespace"))
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=kwargs["top_k"]))
        memory = VectorStoreRetrieverMemory(retriever=retriever)
    elif memory_type == "chroma":
        chroma = Chroma(kwargs["index"], OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]))
        retriever = chroma.as_retriever(search_kwargs=dict(k=kwargs["top_k"]))
        memory = VectorStoreRetrieverMemory(retriever=retriever)
    else:
        raise ValueError(f"Memory {memory_type} is not supported currently.")   
    return MemoryWrapper(memory, conversation_threshold, reasoning_threshold)
