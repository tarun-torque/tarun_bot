# api/bot_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai, types, errors
import os, logging, time, json
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "doctor_bot"

logging.basicConfig(level=logging.INFO)

# ---------------- FastAPI setup ----------------
app = FastAPI(title="TarunBot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

# ---------------- Lazy singletons ----------------
client = None
vectorstore = None
embeddings = None
gemini_client = None

def get_qdrant_client():
    global client
    if client is None:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return client

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = Qdrant(client=get_qdrant_client(), collection_name=COLLECTION_NAME, embeddings=None)
    return vectorstore

def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def get_gemini_client():
    global gemini_client
    if gemini_client is None:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return gemini_client

def get_relevant_docs(query: str, k=3):
    try:
        emb = get_embeddings().embed_query(query)
        return get_vectorstore().similarity_search_by_vector(emb, k=k)
    except Exception as e:
        logging.error(f"Error retrieving documents: {e}")
        return []

def ask_gemini_with_retry(prompt: str, retries=3, delay=2):
    client = get_gemini_client()
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            return resp.text.strip()
        except errors.ServerError as e:
            logging.error(f"Gemini error attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return json.dumps({
                    "greeting": "",
                    "possible_causes": "",
                    "self_care": "",
                    "when_to_see_doctor": "⚠️ System busy. Try again later.",
                    "closing": "Regards, AI Doctor Assistant"
                })

@app.get("/")
def home():
    return {"message": "Hello from TarunBot API!"}

@app.post("/ask")
def ask_bot(query: Query):
    docs = get_relevant_docs(query.question)
    context = "\n".join([d.page_content for d in docs]) if docs else ""

    prompt = f"""
You are an AI Doctor Assistant.
Use context if available.
Answer in JSON format only.

Context:
{context}

User Question: {query.question}
"""

    answer = ask_gemini_with_retry(prompt)

    try:
        return {"answer": json.loads(answer)}
    except json.JSONDecodeError:
        logging.warning("Fallback JSON used")
        return {"answer": {
            "greeting": "Hello!",
            "possible_causes": "",
            "self_care": "",
            "when_to_see_doctor": "⚠️ Unable to process your request. Please consult a doctor.",
            "closing": "Regards, AI Doctor Assistant"
        }}
