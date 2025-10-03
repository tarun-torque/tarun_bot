# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from google import genai
from google.genai import types
import os, time, logging
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import List, Union, Any
import numpy as np

# ---------------- Load environment variables ----------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "doctor_bot")

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Gemini client setup ----------------
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set in environment variables.")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- Helper to extract vector from SDK response ----------------
def _extract_vector_from_embedding_obj(obj: Any) -> List[float]:
    """
    Normalize various shapes returned by google-genai SDK into List[float].
    """
    # direct list/tuple/ndarray
    if isinstance(obj, (list, tuple, np.ndarray)):
        return list(obj)
    # object attributes
    if hasattr(obj, "values"):
        vals = getattr(obj, "values")
        return list(vals)
    if hasattr(obj, "embedding"):
        vals = getattr(obj, "embedding")
        return list(vals)
    # dict-like
    try:
        if isinstance(obj, dict):
            if "values" in obj:
                return list(obj["values"])
            if "embedding" in obj:
                return list(obj["embedding"])
    except Exception:
        pass
    # last resort: try to access first attribute that looks like a sequence
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(obj, attr)
            if isinstance(val, (list, tuple, np.ndarray)):
                return list(val)
        except Exception:
            continue
    raise ValueError("Cannot extract embedding vector from object of type: %s" % type(obj))

# ---------------- LangChain-compatible Gemini embeddings ----------------
class GeminiEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper for google-genai (Gemini).
    """

    def __init__(self, client: genai.Client, model: str = "gemini-embedding-001"):
        self.client = client
        self.model = model

    def _normalize_texts(self, texts: Union[str, List[str], np.ndarray]) -> List[str]:
        if isinstance(texts, str):
            return [texts]
        if isinstance(texts, (list, tuple)):
            return [str(t) for t in texts]
        if isinstance(texts, np.ndarray):
            return [str(t) for t in texts.tolist()]
        return [str(texts)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        normalized = self._normalize_texts(texts)
        logger.info("GeminiEmbeddings.embed_documents: embedding %d documents", len(normalized))
        result = self.client.models.embed_content(
            model=self.model,
            contents=normalized
        )
        vectors = []
        for e in result.embeddings:
            vec = _extract_vector_from_embedding_obj(e)
            vectors.append(vec)
        if vectors:
            logger.info("Received %d embeddings, dim=%d", len(vectors), len(vectors[0]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        normalized = self._normalize_texts(text)
        logger.info("GeminiEmbeddings.embed_query: embedding single query")
        result = self.client.models.embed_content(
            model=self.model,
            contents=normalized
        )
        vec = _extract_vector_from_embedding_obj(result.embeddings[0])
        logger.info("Received query embedding, dim=%d", len(vec))
        return vec

    def __call__(self, texts: Union[str, List[str], np.ndarray]) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            return self.embed_query(texts)
        if isinstance(texts, (list, tuple, np.ndarray)):
            return self.embed_documents(list(texts))
        return self.embed_query(str(texts))


# ---------------- Qdrant setup with auto-check for vector dim ----------------
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embeddings = GeminiEmbeddings(client=gemini_client, model="gemini-embedding-001")

    # Determine the embedding dimension by calling the embedder once
    try:
        sample_vec = embeddings.embed_query("test embedding dimension")
        embedding_dim = len(sample_vec)
        logger.info("Detected embedding dimension from Gemini: %d", embedding_dim)
    except Exception as e:
        logger.exception("Failed to get embedding dimension from Gemini: %s", e)
        raise

    # Try to check existing collection
    try:
        col_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        existing_dim = None
        # col_info may be a model or dict, try to read vectors config defensively
        try:
            # qdrant_client.get_collection returns CollectionInfo with attribute 'vectors'
            vectors_info = getattr(col_info, "vectors", None)
            if vectors_info is None and isinstance(col_info, dict):
                vectors_info = col_info.get("vectors")
            if isinstance(vectors_info, dict):
                existing_dim = vectors_info.get("size")
            else:
                existing_dim = getattr(vectors_info, "size", None)
        except Exception:
            existing_dim = None

        logger.info("Existing Qdrant collection '%s' vector size: %s", COLLECTION_NAME, existing_dim)

        if existing_dim is None:
            # Could not determine existing dimension — recreate to be safe
            logger.warning("Could not determine existing collection vector dimension. Recreating collection with dim=%d", embedding_dim)
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=rest_models.VectorParams(size=embedding_dim, distance=rest_models.Distance.COSINE),
            )
        elif existing_dim != embedding_dim:
            # Dimension mismatch: recreate collection (this deletes existing data!)
            logger.warning(
                "Vector dimension mismatch for collection '%s': existing=%s, expected=%s. "
                "Recreating collection (WARNING: this will DELETE existing collection data).",
                COLLECTION_NAME,
                existing_dim,
                embedding_dim,
            )
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=rest_models.VectorParams(size=embedding_dim, distance=rest_models.Distance.COSINE),
            )
        else:
            logger.info("Collection vector dimension matches embeddings. No recreate needed.")
    except Exception as e:
        # If get_collection failed, try creating the collection
        logger.info("Collection '%s' not found or error reading it. Creating with dim=%d", COLLECTION_NAME, embedding_dim)
        try:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=rest_models.VectorParams(size=embedding_dim, distance=rest_models.Distance.COSINE),
            )
            logger.info("Collection created.")
        except Exception as e2:
            logger.exception("Failed to create collection: %s", e2)
            raise

    # Initialize LangChain Qdrant wrapper
    vectorstore = Qdrant(client=qdrant_client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    logger.info("Qdrant vectorstore initialized with Gemini embeddings (dim=%d).", embedding_dim)
except Exception as e:
    logger.exception("Error initializing Qdrant with Gemini embeddings: %s", e)
    raise


# ---------------- Gemini LLM setup ----------------
def ask_gemini_with_retry(prompt: str, retries=3, delay=2) -> str:
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=20)
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error("Gemini error on attempt %d: %s", attempt + 1, e)
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


# ---------------- Embedding test endpoint ----------------
@app.get("/test-embed")
def test_embed():
    sample_docs = ["What is the meaning of life?", "How do I bake a cake?"]
    sample_query = "What is the purpose of existence?"
    try:
        doc_vecs = embeddings.embed_documents(sample_docs)
        query_vec = embeddings.embed_query(sample_query)
        return {
            "status": "ok",
            "num_doc_embeddings": len(doc_vecs),
            "doc_embedding_dim": len(doc_vecs[0]) if doc_vecs else 0,
            "query_embedding_dim": len(query_vec),
        }
    except Exception as e:
        logger.exception("Embedding test failed: %s", e)
        return {"status": "error", "error": str(e)}


# ---------------- Ask endpoint ----------------
@app.post("/ask")
def ask_bot(query: Query):
    try:
        # Retrieve context from Qdrant (this will use embeddings under the hood)
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
            logger.warning("Gemini did not return valid JSON, using text fallback.")
            answer_json = {
                "greeting": "Hello!",
                "possible_causes": answer,
                "self_care": "",
                "when_to_see_doctor": "",
                "closing": "Regards, AI Doctor Assistant"
            }

        return {"answer": answer_json}

    except Exception as e:
        logger.exception("Error in /ask endpoint: %s", e)
        return {"answer": {
            "greeting": "Hello!",
            "possible_causes": "",
            "self_care": "",
            "when_to_see_doctor": f"⚠️ Internal server error: {e}",
            "closing": "Regards, AI Doctor Assistant"
        }}
