# from langchain_core.tools import tool
# from retriever.qdrant_retriever import QdrantRetriever

# # 1. Initialize your existing Qdrant retriever
# retriever = QdrantRetriever()

# @tool
# def query_knowledge_base(query: str) -> str:
#     """
#     Useful for searching company files, uploaded documents, PDFs, 
#     and specific knowledge base records to answer precise questions.
#     """
#     # 2. Fetch matching document chunks from Qdrant
#     docs = retriever.invoke(query)
    
#     # 3. Format chunks into a single text block for the LLM to read
#     return "\n\n".join([doc.page_content for doc in docs])

# from rag import RAGPipeline
# from tools.base_tool import BaseTool


# class RAGTool(BaseTool):
#     """A tool for answering queries using a vector store-backed RAG pipeline."""

#     def __init__(self):
#         super().__init__(
#             name="rag_search",
#             description=(
#                 "Use this tool to answer factual, abbreviation-based, educational, or document-related questions. "
#                 "It searches internal documents using a vector database. "
#                 "Always try this first before considering external tools like web_search, wikipedia, weather etc."
#             )
#         )
#         self.rag = RAGPipeline()

#     def run(self, query: str) -> str:
#         """Run the RAG pipeline for the given query and return the answer."""
#         if not query or not query.strip():
#             return "❌ Query cannot be empty."
#         try:
#             return self.rag.ask(query)
#         except Exception as e:
#             return f"⚠️ RAG processing failed: {str(e)}"
#     def __str__(self):
        
#         return "ragggggggggggggggg toooooooooooool"


# # === For standalone testing ===
# if __name__ == "__main__":
#     rag_tool = RAGTool()
#     question = "Khalid is generally considered by historians to be one of the most seasoned and " \
#     "accomplished generals in Islamic history, and he is likewise commemorated throughout the Arab world. Islamic tradition credits him with decisive " \
#     "battlefield tactics and effective leadership during the early Muslim conquests."
#     answer = rag_tool.run(question)
#     question1 = "What is Khalid"
#     answer1 = rag_tool.run(question1)
#     print(f"Q: {question1}\nA: {answer1}")
    

from langchain_core.tools import tool
from rag import RAGPipeline

# Initialize once
rag_pipeline = RAGPipeline()

@tool
def rag_search(query: str) -> str:
    """
    Search internal documents stored in Qdrant and answer questions
    using Retrieval-Augmented Generation (RAG).
    Use this before external tools.
    """
    print("##################### calling Tooooooooooooool, ")
    if not query or not query.strip():
        return "Query cannot be empty."

    try:
        return rag_pipeline.ask(query)
    except Exception as e:
        return f"RAG processing failed: {e}"