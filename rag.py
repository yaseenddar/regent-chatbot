import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from rich import print
import warnings
from config import Config
from llm.gemini_llm import GeminiLLM
from memory.chat_memory import MemoryManager
from langchain_core.prompts import ChatPromptTemplate
from retriever.qdrant_retriever import QdrantRetriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
warnings.filterwarnings("ignore", category=DeprecationWarning)


class RAGPipeline:
    def __init__(self):
        self.retriever = QdrantRetriever()
        self.memory = MemoryManager()
        self.llm = GeminiLLM().get_client()

        self.prompt = self._load_prompt(Config.RAG_PROMPT)
        self.qa_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.chain = create_retrieval_chain(self.retriever, self.qa_chain)

    def _load_prompt(self, path: str) -> ChatPromptTemplate:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

        return ChatPromptTemplate.from_messages([
            ("system", "{chat_history}\n\n" + system_prompt),
            ("human", "{input}")
        ])

    def messages_to_string(self, messages: list[BaseMessage]) -> str:
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = "unknown"
            history.append(f"{role}: {msg.content}")
        return "\n".join(history)

    def ask(self, query: str) -> str:
        session_id = Config.SESSION_ID

        # 1. Get conversation history as a raw list of message objects
        history_messages = self.memory.get(session_id)

        # 2. Pass the list directly to the input dictionary (No more messages_to_string!)
        inputs = {
            "input": query,
            "chat_history": history_messages
        }

        # 3. Invoke RAG chain
        response = self.chain.invoke(inputs)
        # response = "response"
        # 4. Extract final answer from the dictionary
        answer = response["answer"]
        # answer = "answer"

        # 5. Save raw interaction objects to memory
        self.memory.add(session_id, HumanMessage(content=query))
        self.memory.add(session_id, AIMessage(content=answer))
        
        # Debug print
        print(f"The memory is Updated: {history_messages}")
        
        # 6. Return the actual variable, not a string literal!
        return answer


if __name__ == "__main__":
    rag = RAGPipeline()
    query1 = "The Full form of the K12HSN is Potassium 12"
    query2 = "What is the full form of K12HSN?"
    query3 = "What does the abbreviation stand for?"
    
    response1 = rag.ask(query1)
    # print(f"Q1: {query1}\nA1: {response1}")

    response2 = rag.ask(query2)
    print(f"Q1: {query1}\nA1: {response2}")

    # response3 = rag.ask(query3)
    # print(f"Q2: {query2}\nA2: {response3}")