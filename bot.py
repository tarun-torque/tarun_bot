# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types, errors
import os, time, logging
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import json

# ---------------- Load environment variables ----------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "doctor_bot"

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)

# ---------------- Qdrant setup ----------------
try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    logging.error("Error initializing Qdrant:", e)
    raise e

# ---------------- Gemini setup ----------------
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini_with_retry(prompt: str, retries=3, delay=2) -> str:
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=20)  # increased budget
                ),
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Gemini error on attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return "⚠️ System busy. Please try again later."

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

# ---------------- Home endpoint ----------------
@app.get("/")
def home():
    return {"message": "Hello from TarunBot API!"}

# ---------------- Ask endpoint ----------------
@app.post("/ask")
def ask_bot(query: Query):
    try:
        # Retrieve context
        docs = retriever.get_relevant_documents(query.question)
        context = "\n".join([d.page_content for d in docs]) if docs else ""

        # Prepare Gemini prompt
        gemini_prompt = f"""
You are an AI Doctor Assistant developed by Tarun.
Answer the user’s medical question in a safe, non-prescriptive way.
You may mention general medicine classes (OTC only) but not exact dosages.
Use context if available.

JSON format recommendation:
{{
    "greeting": "short friendly greeting",
    "possible_causes": "possible causes or context-based info",
    "self_care": "safe self-care advice including OTC medicine classes only",
    "when_to_see_doctor": "urgent instructions or red flags",
    "closing": "Regards, AI Doctor Assistant"
}}

Context:
{context}

User Question: {query.question}

Answer in JSON format if possible. If not, just give text.
"""

        answer = ask_gemini_with_retry(gemini_prompt)

        # Try parsing JSON, fallback if invalid
        try:
            answer_json = json.loads(answer)
        except json.JSONDecodeError:
            logging.warning("Gemini did not return valid JSON, using text fallback.")
            answer_json = {
                "greeting": "Hello!",
                "possible_causes": answer,
                "self_care": "",
                "when_to_see_doctor": "",
                "closing": "Regards, AI Doctor Assistant"
            }

        return {"answer": answer_json}

    except Exception as e:
        logging.error("Error in /ask endpoint:", e)
        return {"answer": {
            "greeting": "Hello!",
            "possible_causes": "",
            "self_care": "",
            "when_to_see_doctor": f"⚠️ Internal server error: {e}",
            "closing": "Regards, AI Doctor Assistant"
        }}
