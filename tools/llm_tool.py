import os
from config import Config
from tools.base_tool import BaseTool
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
class LLMInstructionTol(BaseTool):

    def __init__(self):
        super().__init__(
            name="llm_instruction",
            description=(
                "Handles creatice and instructional tasks using an LLM."
                "Use this tool fot tasks like generating text,poem generation,storytelling, answers, summarizing, and other creative or instructional tasks."
                "when no specific tool is suitable for the task, this tool can be used to handle general LLM tasks."

            )
        )
        self.llm = ChatGoogleGenerativeAI(
            google_api_key = os.getenv("GOOGLE_API_KEY"),
            model = Config.LLM_MODEL,
            temperature = Config.TEMPERATURE
        )

    def run(self,query:str) -> str:
        
        if not query or not query.strip():
            return "Error: Query cannot be empty"
        try:
            response = self.llm.invoke([HumanMessage(content=query)])
            return response.content.strip()
        except Exception as e:
            return f"Error processing the query: {str(e)}"
    def __str__(self):
        return f"Tool Name: {self.name}\nDescription: {self.description}"
# for standalone testing 
if __name__ == "__main__":
    llm_tool = LLMInstructionTol()
    test_query = "Write a short poetry about the mountain life"
    result = llm_tool.run(test_query)
    print(f"##### LLm Response {result}")