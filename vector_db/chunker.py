import hashlib
from typing import List
from config import Config
from utils.normalizer import Normalizer
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich import print

class DocumentChunker:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.existing_hashes = set()  # To track existing hashes and avoid duplicates
        self.normalizer = Normalizer()


    def hash_text(self,text:str) -> str:
        """Generate a unique hash for the given text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def split_documents(self,docs:List[Document]) -> List[dict]:
        """Split and duplicacte documents. Returns list of dicts with id, text, metadata."""
        chunks = self.splitter.split_documents(docs)
        result = []

        for i, chunk in enumerate(chunks):
            normalized_text = self.normalizer.normalize_text(chunk.page_content)

            if not normalized_text:
                continue
            chunk_hash = self.hash_text(normalized_text)
            if chunk_hash in self.existing_hashes:
                continue
            self.existing_hashes.add(chunk_hash)

            result.append({
                "id":int(chunk_hash,16) % (10 ** 9),
                "text":normalized_text,
                "metadata":{
                    **chunk.metadata,
                    "chunk_order":i # Preserve order
                }
            })
        return result

if __name__ == "__main__":
    sample_doc = [
        Document(
            page_content=
            """
This file is the qdrant_retriever.py module located in the retriever/ folder of your project. It is a core component for the RAG (Retrieval-Augmented Generation) pipeline.

Here is exactly what this file is doing:

1. Creating a Custom LangChain Retriever
It defines a class called QdrantRetriever which inherits from LangChain's native BaseRetriever. By subclassing BaseRetriever, this script integrates seamlessly into any LangChain expression language (LCEL) chain or agent workflow.

2. Connecting to the Qdrant Vector Database
In the __init__ constructor, it initializes a client connection to your vector database using QdrantDBClient() (which is imported from your local vector_db.qdrant_db module).

It pulls a configuration setting (Config.TOP_K) to determine exactly how many documents (_k) it should fetch during a search.

3. Executing the Semantic Search
It overrides LangChain's internal _get_relevant_documents method. When the agent passes a natural language string query into this retriever, this method:

Calls the database search function: self._qdrant_client.search(query=input, top_k=self._k)

Queries your Qdrant instance to look up the most contextually relevant chunks of text.

Returns them as a list of LangChain Document objects.

4. Self-Testing Block
At the very bottom, the if __name__ == "__main__": block allows you to run this file directly in your terminal to test if your database connection works. It attempts to query "Who is the president of the United States?" and prints out previews of the top matching documents stored in your local Qdrant database.

To answer your first question: Are there any built-in chatbots?
No, this file does not contain a chatbot user interface or an LLM orchestration loop. It is strictly a "data provider" component.

To turn this into a chatbot, you will need to import this QdrantRetriever class into your agent or chain script (like your main.py or graph.py), convert it into a tool, and pass it to your LLM.            
""",
            metadata={"source":"example.txt"}
        )
    ]

    chunker = DocumentChunker()
    chunker = chunker.split_documents(sample_doc)

    for i,chunk in enumerate(chunker):
        print(f"####### Chunk {i}\n{chunker[i]}")
