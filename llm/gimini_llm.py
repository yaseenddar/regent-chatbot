import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import Config

load_dotenv()

class GeminiLLM:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in the environment variables")
        
        self.model_name = Config.LLM_MODEL
        self.temperature = Config.TEMPERATURE
        self.gemini_client = self._initialize_client()

    def _initialize_client(self):
        return ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model_name,
            temperature=self.temperature 
        )
    def get_client(self):
        return self.gemini_client
    
if __name__ == "__main__":
    gemini_llm = GeminiLLM()
    llm = gemini_llm.get_client()
    response = llm.invoke([HumanMessage(content="Explain LangChain in 5 sentences.")])
    print("### Gemini Response ### ",response.content) 