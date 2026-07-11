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
load_dotenv()

class Agent:
    def __init__(self):
        # 1. Load your system prompt
       
        system_prompt = """
    You are an AI assistant with tools.

    Format every response exactly as:

    Thought: <reasoning>
    Action: {"action":"tool_name","action_input":"input"}
    Observation: <tool result>

    Repeat as needed, then end with:

    Final Answer: <answer>

    Rules:
    - After every Observation, output only: Thought, Action, or Final Answer.
    - Never give conclusions without a prefix.
    - Action must be a single-line JSON object.
    - Prefer tools over guessing; never invent facts.
    - For factual, abbreviation, or document queries, try `rag_search` first.
    - If `rag_search` is insufficient, use `web_search`, `wikipedia`, `weather`, etc.
    - Use `calculator` for math; convert natural language math to Python syntax (e.g., "2 to the power 5" → `2**5`).
    - Use `llm_instruction` for summarization, rewriting, explanations, or creative tasks.
    - If a tool fails, continue reasoning and try another tool.
    - For time-sensitive information, validate dates against today's date and flag future-dated results as uncertain.
    - If no tool provides enough information, reply:
    Final Answer: I couldn’t find enough information.

    Available tools:
    rag_search, web_search, wikipedia, weather, calculator, llm_instruction
    """
        
        # 2. Modern Chat Prompt Template layout
       # Add template="jinja2" to prevent curly-brace collision with standard strings
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 3. Get LLM client natively
        self.llm = GeminiLLM().get_client()
        
        # 4. Dynamically load all tools
        # registry = ToolRegistry()
        self.tools = [calculator]
        
        # 5. Create the modern Tool Calling Agent
        # This utilizes Gemini's native API tool-calling, eliminating parser bugs
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        # 6. Create the Executor executor
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True
        )

    def load_prompt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def run(self, query: str, history: list[BaseMessage] = None) -> str:
        # Format variables to match the prompt template expectation
        chat_history = history if history else []
        
        # Invoke the modern executor
        response = self.agent_executor.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        return response["output"]

if __name__ == "__main__":
    agent = Agent()
    user_query = "3 +4 and 5 * 4"
    answer = agent.run(user_query)
    print("\n### Agent Response:\n", answer)