import os
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from pinecone import Pinecone

# ==================== LOAD ENVIRONMENT VARIABLES ====================
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="SQL RAG Assistant", page_icon="🧠")
st.title("🧠 SQL RAG Assistant (Groq + RAG)")

# ==================== LOAD AND SPLIT PDF ====================
st.write("📄 Loading SQL Manual PDF...")
loader = PyPDFLoader("SQL-Manual.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)
st.success(f"✅ Loaded and split into {len(docs)} chunks")

# ==================== PINECONE SETUP ====================
embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pdf-chat-index"

vectorstore = PineconeVectorStore.from_documents(docs, embedding=embeddings, index_name=index_name)
st.success("✅ Vector store connected successfully")

# ==================== LLM SETUP ====================
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
st.success("🤖 Connected to Groq LLM")

# ==================== RAG FUNCTION ====================
def get_answer(query: str) -> str:
    try:
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([r.page_content for r in results])

        prompt = f"""
You are a helpful SQL tutor. Use the given context and your SQL knowledge.
Explain the answer clearly and give short examples if useful.

Context:
{context}

Question:
{query}

Answer:
"""
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"⚠️ Error processing query: {str(e)}"


# ==================== HTTP SERVER HANDLER ====================
class RAGRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="application/json"):
        """Helper to send common headers."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self._set_headers()

    def do_POST(self):
        """Handle POST /query requests."""
        if self.path == "/query":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body)
                query = data.get("query", "")

                answer = get_answer(query)
                self._set_headers()
                self.wfile.write(json.dumps({"answer": answer}).encode())
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found."}).encode())


# ==================== SERVER THREAD ====================
def run_server():
    port = int(os.environ.get("PORT", 8502))  # Render assigns PORT dynamically
    server_address = ('', port)
    httpd = HTTPServer(server_address, RAGRequestHandler)
    print(f"✅ API running at http://0.0.0.0:{port}/query")
    httpd.serve_forever()


threading.Thread(target=run_server, daemon=True).start()
st.info("✅ Backend API live (Render-compatible with CORS)")

# ==================== STREAMLIT INTERFACE ====================
query = st.text_input("Ask your SQL question:")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            answer = get_answer(query)
            st.markdown(answer)
