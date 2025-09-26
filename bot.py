# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types, errors
import os, time
from dotenv import load_dotenv

# ---------------- Load environment variables ----------------
load_dotenv()
QDRANT_URL = "https://77515254-76b4-47f6-9d4c-cf5417a3daba.eu-central-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "doctor_bot"

# ---------------- Qdrant setup ----------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ---------------- Gemini setup ----------------
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini_with_retry(prompt: str, retries=3, delay=2) -> str:
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            return response.text
        except errors.ServerError as e:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return "Sorry, the model is overloaded. Try later."

# ---------------- FastAPI setup ----------------
info='If user asked about who are you , who made you or similar question or ask about your work answer i am AI Doctor assistant developed by Tarun because are very helpful for medical information'
rules= "Greeting the user first and the end say regards AI doctor assistant ,Use the following context to answer and return the conversation from context and do not include the doctor name and address mobile number, if the information not present in the context then do not say that information is not present"

app = FastAPI(title="TarunBot API")

class Query(BaseModel):
    question: str

# Home endpoint
@app.get("/")
def home():
    return {"message": "Hello from TarunBot API!"}

# Ask endpoint
@app.post("/ask")
def ask_bot(query: Query):
    docs = retriever.get_relevant_documents(query.question)
    if docs:
        context = "\n".join([d.page_content for d in docs])
        gemini_prompt = f"{info} {rules}, \n\n{context}\n\nQuestion: {query.question}"
    else:
        gemini_prompt = query.question

    answer = ask_gemini_with_retry(gemini_prompt)
    return {"answer": answer}
