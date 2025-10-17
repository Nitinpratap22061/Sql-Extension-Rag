import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
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

# ==================== FLASK APP SETUP ====================
app = Flask(__name__)
CORS(app)  # ✅ Allow all origins for your Chrome extension

# ==================== LOAD AND SPLIT PDF ====================
print("📄 Loading SQL Manual PDF...")
loader = PyPDFLoader("SQL-Manual.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)
print(f"✅ Loaded and split into {len(docs)} chunks")

# ==================== PINECONE SETUP ====================
embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pdf-chat-index"

vectorstore = PineconeVectorStore.from_documents(docs, embedding=embeddings, index_name=index_name)
print("✅ Vector store connected successfully")

# ==================== LLM SETUP ====================
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
print("🤖 Connected to Groq LLM")

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

# ==================== API ROUTE ====================
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query_text = data.get("query", "")
    answer = get_answer(query_text)
    return jsonify({"answer": answer})

# ==================== RUN FLASK ====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))  # Render sets PORT env variable
    print(f"🚀 API running on port {port}")
    app.run(host="0.0.0.0", port=port)
