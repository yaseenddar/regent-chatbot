import os
from config import Config
from dotenv import load_dotenv
from llm.gemini_llm import GeminiLLM
from tool_registry import ToolRegistry
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages import SystemMessage
from tools.calculator import calculator
from tools.rag_tool import rag_search

load_dotenv()

class Agent:
    def __init__(self):
        # 1. Simplified System Prompt (Removed manual ReAct formatting rules)
        system_prompt = """You are a helpful AI assistant equipped with tools to assist you.

            Rules:
            - Prefer using tools over guessing; never invent facts.
            - For factual queries, abbreviations, or internal document lookups, try `rag_search` first.
            - Use `calculator` for math calculations. Convert natural language math expressions into clear mathematical notation if needed.
            - For time-sensitive information, validate dates against today's date.
            - If no tool provides enough information, tell the user honestly that you couldn't find enough information.
            """
        
        # 2. Modern Chat Prompt Template layout
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 3. Get LLM client natively
        self.llm = GeminiLLM().get_client()
        
        # 4. Dynamically load all tools
        self.tools = [calculator, rag_search]
        
        # 5. Create the modern Tool Calling Agent
        # Gemini handles tool calls natively via API now instead of parsing text
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        # 6. Create the Executor
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True
        )

    def load_prompt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def run(self, query: str, history: list[BaseMessage] = None) -> str:
        chat_history = history if history else []
        
        response = self.agent_executor.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        return response["output"]

if __name__ == "__main__":
    agent = Agent()
    user_query = "find me the projects in the resume uploaded just the names"
    answer = agent.run(user_query)
    print("\n### Agent Response:\n", answer)