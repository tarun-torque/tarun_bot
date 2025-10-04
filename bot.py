# api.py
import os
import time
import json
import logging
import threading
import asyncio
from typing import List, Union, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------- Load environment variables ----------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "doctor_bot")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))
ENABLE_TRACEMALLOC = os.getenv("ENABLE_TRACEMALLOC", "0") == "1"
TRACEMALLOC_LIMIT = int(os.getenv("TRACEMALLOC_LIMIT_BYTES", str(25 * 1024 * 1024)))  # default 25MB

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tarunbot")

# ---------------- Module-level cache + locks ----------------
_init_lock = threading.Lock()
_services_initialized = False
_services = {
    "gemini_client": None,
    "qdrant_client": None,
    "embeddings": None,
    "vectorstore": None,
    "retriever": None,
    "embedding_dim": None,
    "semaphore": asyncio.Semaphore(MAX_CONCURRENT_REQUESTS),
    "tracemalloc_enabled": False,
}

# ---------------- Helper: extract vector ----------------
def _extract_vector_from_embedding_obj(obj: Any) -> List[float]:
    """
    Normalize various shapes returned by google-genai SDK into List[float].
    """
    # direct list/tuple
    if isinstance(obj, (list, tuple)):
        return list(obj)
    # numpy arrays (import only if needed)
    try:
        import numpy as _np  # local import to avoid global heavy imports
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # object attributes
    if hasattr(obj, "values"):
        vals = getattr(obj, "values")
        if isinstance(vals, (list, tuple)):
            return list(vals)
    if hasattr(obj, "embedding"):
        vals = getattr(obj, "embedding")
        if isinstance(vals, (list, tuple)):
            return list(vals)
    # dict-like
    if isinstance(obj, dict):
        if "values" in obj:
            return list(obj["values"])
        if "embedding" in obj:
            return list(obj["embedding"])
    # last resort: try to find first attr that's a sequence
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(obj, attr)
            if isinstance(val, (list, tuple)):
                return list(val)
        except Exception:
            continue
    raise ValueError("Cannot extract embedding vector from object of type: %s" % type(obj))


# ---------------- LangChain-compatible Gemini embeddings (lazy use) ----------------
class GeminiEmbeddings:
    """
    Minimal LangChain-like embeddings wrapper for google-genai (Gemini).
    This class intentionally avoids heavy imports at module import time.
    """

    def __init__(self, client, model: str = EMBED_MODEL, batch_size: int = EMBED_BATCH_SIZE):
        self.client = client
        self.model = model
        self.batch_size = max(1, int(batch_size))

    def _normalize_texts(self, texts: Union[str, List[str]]) -> List[str]:
        if isinstance(texts, str):
            return [texts]
        if isinstance(texts, (list, tuple)):
            return [str(t) for t in texts]
        return [str(texts)]

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        result = self.client.models.embed_content(model=self.model, contents=batch)
        vectors = []
        for e in result.embeddings:
            vec = _extract_vector_from_embedding_obj(e)
            vectors.append(vec)
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        normalized = self._normalize_texts(texts)
        all_vectors = []
        for i in range(0, len(normalized), self.batch_size):
            batch = normalized[i : i + self.batch_size]
            logger.debug("Embedding batch size=%d", len(batch))
            vectors = self._embed_batch(batch)
            all_vectors.extend(vectors)
        logger.info("embed_documents: produced %d vectors", len(all_vectors))
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        normalized = self._normalize_texts(text)[0]
        result = self.client.models.embed_content(model=self.model, contents=[normalized])
        vec = _extract_vector_from_embedding_obj(result.embeddings[0])
        return vec

    def __call__(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_documents(list(texts))


# ---------------- Optional tracemalloc enable (lazy) ----------------
def _maybe_start_tracemalloc():
    if ENABLE_TRACEMALLOC and not _services.get("tracemalloc_enabled", False):
        try:
            import tracemalloc as _tracemalloc
            _tracemalloc.start()
            _services["tracemalloc_enabled"] = True
            logger.info("tracemalloc started (ENABLE_TRACEMALLOC=1).")
        except Exception as e:
            logger.exception("Failed to start tracemalloc: %s", e)
            _services["tracemalloc_enabled"] = False


# ---------------- Lazy initialization ----------------
def initialize_services():
    """
    Lazily initialize gemini client, qdrant client, embeddings, vectorstore, retriever.
    Safe to call multiple times; protected by lock.
    """
    global _services_initialized, _services

    if _services_initialized:
        return

    with _init_lock:
        if _services_initialized:
            return

        # Possibly start tracemalloc (only if env says so)
        if ENABLE_TRACEMALLOC:
            _maybe_start_tracemalloc()

        # ---------- Gemini client ----------
        try:
            if GEMINI_API_KEY:
                from google import genai
                gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                logger.info("Gemini client initialized.")
            else:
                gemini_client = None
                logger.warning("GEMINI_API_KEY not set; Gemini client disabled.")
            _services["gemini_client"] = gemini_client
        except Exception as e:
            logger.exception("Failed to initialize Gemini client: %s", e)
            _services["gemini_client"] = None

        # ---------- Qdrant client and vectorstore ----------
        lc_available = False
        if QDRANT_URL:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http import models as rest_models
                # Lazy check for LangChain's Qdrant wrapper
                try:
                    from langchain_community.vectorstores import Qdrant as LCQdrant
                    lc_available = True
                except Exception:
                    lc_available = False

                qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
                _services["qdrant_client"] = qdrant_client
                logger.info("Qdrant client initialized at %s", QDRANT_URL)
            except Exception as e:
                logger.exception("Failed to initialize Qdrant client: %s", e)
                _services["qdrant_client"] = None
                lc_available = False
        else:
            logger.warning("QDRANT_URL not set; Qdrant disabled.")
            _services["qdrant_client"] = None
            lc_available = False

        # ---------- embeddings ----------
        if _services["gemini_client"]:
            try:
                embeddings = GeminiEmbeddings(client=_services["gemini_client"], model=EMBED_MODEL)
                _services["embeddings"] = embeddings
                # try to detect dimension (best-effort).
                try:
                    sample_vec = embeddings.embed_query("test embedding dimension")
                    _services["embedding_dim"] = len(sample_vec)
                    logger.info("Detected embedding dim: %d", _services["embedding_dim"])
                except Exception as e:
                    logger.warning("Could not detect embedding dimension automatically: %s. Proceeding without forcing collection recreation.", e)
                    _services["embedding_dim"] = None
            except Exception as e:
                logger.exception("Failed to create embeddings wrapper: %s", e)
                _services["embeddings"] = None
                _services["embedding_dim"] = None
        else:
            _services["embeddings"] = None
            _services["embedding_dim"] = None

        # ---------- collection handling (defensive, non-destructive) ----------
        if _services["qdrant_client"] and _services["embeddings"]:
            try:
                from qdrant_client.http import models as rest_models
                # get collection info if exists
                try:
                    col_info = _services["qdrant_client"].get_collection(collection_name=COLLECTION_NAME)
                    existing_dim = None
                    vectors_info = getattr(col_info, "vectors", None)
                    if vectors_info is None and isinstance(col_info, dict):
                        vectors_info = col_info.get("vectors")
                    if isinstance(vectors_info, dict):
                        existing_dim = vectors_info.get("size")
                    else:
                        existing_dim = getattr(vectors_info, "size", None)
                    logger.info("Found existing collection '%s' vector size: %s", COLLECTION_NAME, existing_dim)
                except Exception:
                    col_info = None
                    existing_dim = None

                # If collection not found, create it
                if col_info is None:
                    if _services["embedding_dim"]:
                        try:
                            _services["qdrant_client"].create_collection(
                                collection_name=COLLECTION_NAME,
                                vectors_config=rest_models.VectorParams(size=_services["embedding_dim"], distance=rest_models.Distance.COSINE),
                            )
                            logger.info("Created Qdrant collection '%s' with dim=%s", COLLECTION_NAME, _services["embedding_dim"])
                        except Exception as e:
                            logger.exception("Failed to create collection '%s': %s", COLLECTION_NAME, e)
                    else:
                        # If we don't know embedding dim, create with a safe default (common dims: 1536)
                        DEFAULT_DIM = int(os.getenv("DEFAULT_EMBED_DIM", "1536"))
                        try:
                            _services["qdrant_client"].create_collection(
                                collection_name=COLLECTION_NAME,
                                vectors_config=rest_models.VectorParams(size=DEFAULT_DIM, distance=rest_models.Distance.COSINE),
                            )
                            logger.info("Created Qdrant collection '%s' with default dim=%d", COLLECTION_NAME, DEFAULT_DIM)
                            if _services["embedding_dim"] is None:
                                _services["embedding_dim"] = DEFAULT_DIM
                        except Exception as e:
                            logger.exception("Failed to create collection with default dim: %s", e)
                else:
                    # If collection exists and dims mismatch, do not delete/recreate automatically (destructive).
                    if existing_dim and _services["embedding_dim"] and existing_dim != _services["embedding_dim"]:
                        logger.warning(
                            "Existing collection dimension (%s) differs from detected embedding dimension (%s). "
                            "Will use existing collection without destructive recreate. If you want to recreate, do it manually.",
                            existing_dim,
                            _services["embedding_dim"],
                        )

                # Initialize LangChain Qdrant wrapper if LC is available
                if lc_available:
                    try:
                        from langchain_community.vectorstores import Qdrant as LCQdrant
                        vectorstore = LCQdrant(client=_services["qdrant_client"], collection_name=COLLECTION_NAME, embeddings=_services["embeddings"])
                        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
                        _services["vectorstore"] = vectorstore
                        _services["retriever"] = retriever
                        logger.info("LangChain Qdrant vectorstore initialized.")
                    except Exception as e:
                        logger.exception("Failed to initialize LangChain Qdrant wrapper: %s", e)
                        _services["vectorstore"] = None
                        _services["retriever"] = None
                else:
                    _services["vectorstore"] = None
                    _services["retriever"] = None

            except Exception as e:
                logger.exception("Error during Qdrant/collection handling: %s", e)
        else:
            logger.info("Qdrant or Embeddings not available; skipping vectorstore initialization.")

        _services_initialized = True
        logger.info("Service initialization complete.")


# ---------------- Gemini LLM helper with retry ----------------
def ask_gemini_with_retry(prompt: str, retries: int = 2, delay: float = 1.0, thinking_budget: int = 10) -> str:
    """
    Best-effort wrapper to call Gemini LLM. Returns a plain text result or an error message.
    """
    initialize_services()
    client = _services.get("gemini_client")
    if not client:
        logger.warning("Gemini client not configured; returning fallback message.")
        return "⚠️ LLM not configured (GEMINI_API_KEY missing)."

    from google.genai import types

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
                ),
            )
            text = getattr(response, "text", None) or getattr(response, "result", None) or str(response)
            return text.strip()
        except Exception as e:
            logger.error("Gemini error on attempt %d: %s", attempt + 1, e)
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                return "⚠️ System busy. Please try again later."


# ---------------- FastAPI app ----------------
app = FastAPI(title="TarunBot API (optimized)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


@app.on_event("startup")
async def on_startup():
    """
    Keep startup light on serverless hosts. Services initialize lazily on first request.
    """
    logger.info("TarunBot API started. Services will initialize lazily on first use.")


@app.get("/")
def home():
    return {"message": "Hello from TarunBot API (optimized)!"}


@app.get("/test-embed")
def test_embed():
    """
    Quick endpoint to test embeddings. Initializes services lazily.
    """
    initialize_services()
    emb = _services.get("embeddings")
    if not emb:
        return {"status": "error", "error": "Embeddings not configured (GEMINI_API_KEY missing or init failed)."}
    try:
        sample_docs = ["What is the meaning of life?", "How do I bake a cake?"]
        sample_query = "What is the purpose of existence?"
        doc_vecs = emb.embed_documents(sample_docs)
        query_vec = emb.embed_query(sample_query)
        return {
            "status": "ok",
            "num_doc_embeddings": len(doc_vecs),
            "doc_embedding_dim": len(doc_vecs[0]) if doc_vecs else 0,
            "query_embedding_dim": len(query_vec),
        }
    except Exception as e:
        logger.exception("Embedding test failed: %s", e)
        return {"status": "error", "error": str(e)}


@app.post("/ask")
async def ask_bot(query: Query):
    """
    Main ask endpoint:
    - Limits concurrent requests via a semaphore to avoid memory spikes.
    - Lazily initializes services.
    - Uses retriever (if available) to fetch context and calls Gemini LLM.
    - Returns JSON response parsed from LLM when possible, otherwise a fallback structure.
    """
    initialize_services()

    sem: asyncio.Semaphore = _services.get("semaphore", asyncio.Semaphore(MAX_CONCURRENT_REQUESTS))
    await sem.acquire()
    try:
        retriever = _services.get("retriever")
        context = ""
        if retriever:
            try:
                docs = []
                try:
                    docs = retriever.get_relevant_documents(query.question)
                except Exception:
                    try:
                        docs = retriever.retrieve(query.question)
                    except Exception:
                        docs = []
                context = "\n".join([getattr(d, "page_content", str(d)) for d in docs]) if docs else ""
            except Exception as e:
                logger.exception("Retriever error: %s", e)
                context = ""

        gemini_prompt = f"""
You are an AI Doctor Assistant developed by Tarun.
Answer the user’s medical question in a safe, non-prescriptive way.
You may mention general medicine classes (OTC only) but not exact dosages.
Use context if available.

Provide output in JSON like:
{{
  "greeting": "...",
  "possible_causes": "...",
  "self_care": "...",
  "when_to_see_doctor": "...",
  "closing": "Regards, AI Doctor Assistant"
}}

Context:
{context}

User Question: {query.question}
"""

        raw_answer = ask_gemini_with_retry(gemini_prompt)

        try:
            answer_json = json.loads(raw_answer)
        except Exception:
            logger.warning("LLM did not return valid JSON. Returning fallback JSON with raw text.")
            answer_json = {
                "greeting": "Hello!",
                "possible_causes": raw_answer,
                "self_care": "",
                "when_to_see_doctor": "",
                "closing": "Regards, AI Doctor Assistant",
            }

        return {"answer": answer_json}
    except Exception as e:
        logger.exception("Error in /ask endpoint: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        sem.release()


# ---------------- Memory endpoint ----------------
@app.get("/memory")
def memory(include_tracemalloc: bool = False):
    """
    Returns memory metrics for the running process.
    - include_tracemalloc=True returns top tracemalloc stats if tracemalloc is enabled via ENABLE_TRACEMALLOC=1.
    - This endpoint lazy-imports psutil and tracemalloc to avoid adding overhead unless requested.
    """
    # lazy import psutil
    try:
        import psutil
    except Exception as e:
        logger.exception("psutil is required for /memory endpoint but not installed: %s", e)
        return {
            "error": "psutil not installed. Install with `pip install psutil` to use this endpoint."
        }

    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    rss = getattr(mem, "rss", None)
    vms = getattr(mem, "vms", None)
    percent = round(proc.memory_percent(), 4)
    num_threads = proc.num_threads()
    num_fds = None
    try:
        num_fds = proc.num_fds()
    except Exception:
        num_fds = None

    resp = {
        "pid": proc.pid,
        "rss_bytes": rss,
        "rss_mb": round(rss / (1024**2), 2) if rss is not None else None,
        "vms_bytes": vms,
        "vms_mb": round(vms / (1024**2), 2) if vms is not None else None,
        "mem_percent": percent,
        "threads": num_threads,
        "open_fds": num_fds,
        "services_initialized": _services_initialized,
        "enabled_tracemalloc_env": ENABLE_TRACEMALLOC,
        "tracemalloc_running": _services.get("tracemalloc_enabled", False),
    }

    # Optionally include top tracemalloc stats if enabled and requested
    if include_tracemalloc and ENABLE_TRACEMALLOC and _services.get("tracemalloc_enabled", False):
        try:
            import tracemalloc
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")[:10]
            top = []
            for stat in top_stats:
                # format trace to something readable
                tb = stat.traceback
                frames = []
                for frame in tb[:3]:  # include up to 3 frames for brevity
                    frames.append({
                        "filename": frame.filename,
                        "lineno": frame.lineno,
                        "name": frame.name,
                    })
                top.append({
                    "size_bytes": stat.size,
                    "size_kb": round(stat.size / 1024, 2),
                    "count": stat.count,
                    "frames": frames,
                })
            resp["tracemalloc_top"] = top
        except Exception as e:
            logger.exception("Failed to collect tracemalloc stats: %s", e)
            resp["tracemalloc_error"] = str(e)
    else:
        if include_tracemalloc and not ENABLE_TRACEMALLOC:
            resp["tracemalloc_error"] = "tracemalloc not enabled; set ENABLE_TRACEMALLOC=1 to enable."

    return resp
