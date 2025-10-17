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

# ==================== API ENDPOINT ====================
class RAGRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/query":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            query = data.get("query", "")

            answer = get_answer(query)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"answer": answer}).encode())
        else:
            self.send_error(404, "Endpoint not found.")

def run_server():
    server_address = ('', 8502)
    httpd = HTTPServer(server_address, RAGRequestHandler)
    print("✅ API running at http://localhost:8502/query")
    httpd.serve_forever()

# ==================== START SERVER THREAD ====================
threading.Thread(target=run_server, daemon=True).start()
st.info("✅ Backend API is live at: http://localhost:8502/query")

# ==================== STREAMLIT INTERFACE ====================
query = st.text_input("Ask your SQL question:")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            answer = get_answer(query)
            st.markdown(answer)
