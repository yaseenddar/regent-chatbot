import os
import streamlit as st
from agent import Agent
from config import Config
from memory.chat_memory import MemoryManager
from vector_db.qdrant_db import QdrantDBClient
from langchain_core.messages import HumanMessage, AIMessage

# Keep tokenizers quiet
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache_resource
def init_backend():
    return {
        "agent": Agent(),
        "memory": MemoryManager(),
        "qdrant": QdrantDBClient()
    }

backend = init_backend()
agent = backend["agent"]
memory = backend["memory"]
qdrant_client = backend["qdrant"]
session_id = Config.SESSION_ID


# Page configuration
st.set_page_config(page_title="RAGent Chatbot", page_icon="💬", layout="wide")
st.title("💬 RAGent Chatbot")

# --- SIDEBAR: FILE UPLOADER ---
with st.sidebar:
    st.markdown("### 📂 Drag & Drop Files Below")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=Config.FILE_EXTENSIONS,
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if st.button("Upload Files", use_container_width=True):
        if not uploaded_files:
            st.error("Please select at least one file first.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            total = len(uploaded_files)
            failed_files = []
            
            for i, file in enumerate(uploaded_files):
                status_text.markdown(f"📄 Processed {i}/{total} file(s)...")
                
                # Streamlit keeps files in memory; we save temporarily or process bytes if supported
                # For this step, assuming qdrant_client works with paths, write to temp file:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    file_chunks = qdrant_client.load_and_chunk_docs(tmp_path)
                    qdrant_client.insert_chunks(file_chunks)
                    print(f"################### {file_chunks}")
                except Exception as e:
                    failed_files.append(file.name)
                finally:
                    os.remove(tmp_path)
                
                progress_bar.progress(int((i + 1) / total * 100))
            
            # Show final summary status
            success_count = total - len(failed_files)
            status_text.success(f"✅ {success_count}/{total} file(s) processed and stored!")
            if failed_files:
                st.warning(f"⚠️ Failed to process:\n" + "\n".join(f"- {f}" for f in failed_files))

# --- MAIN SPACE: CHAT INTERFACE ---
st.markdown("### 🤖 Ask Your Question")

# Fetch history to display past turns on screen refresh
past_messages = memory.get(session_id)
for msg in past_messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Accept new user query input
if query := st.chat_input("Type your question here..."):
    # Immediately render the human message
    with st.chat_message("user"):
        st.write(query)
        
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Fetch latest snapshot of memory right before generation run
            current_history = memory.get(session_id)
            response = agent.run(query, current_history)
            
            # Normalize dictionary or text output
            if isinstance(response, dict) and "output" in response:
                answer = response["output"]
            else:
                answer = str(response)
                
            st.write(answer)
            
    # Commit the exchange to memory backend
    memory.add(session_id, HumanMessage(content=query))
    memory.add(session_id, AIMessage(content=answer))