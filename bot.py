# api.py
"""
Render-safe TarunBot API with guaranteed /ask responses.

Behavior:
- Lazy initialization of Gemini and Qdrant clients.
- Blocking SDK calls run in a thread via asyncio.to_thread.
- Strict concurrency via asyncio.Semaphore (env: MAX_CONCURRENT_REQUESTS).
- If concurrency slot is unavailable immediately, return cached answer (if present) or a safe fallback JSON.
- Caches last N answers (LRU) to increase chance of returning useful content when busy.
- Provides /healthz and /memory endpoints.
"""
import os
import time
import json
import logging
import threading
import asyncio
from typing import List, Union, Any
from collections import OrderedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------- Config ----------------
load_dotenv()
PORT = int(os.getenv("PORT", os.getenv("RENDER_PORT", "10000")))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "doctor_bot")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))  # keep 1 on tiny instances
SEMAPHORE_WAIT_SECONDS = float(os.getenv("SEMAPHORE_WAIT_SECONDS", "0.001"))  # immediate attempt
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "1"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))

CACHE_SIZE = int(os.getenv("ANSWER_CACHE_SIZE", "128"))  # LRU cache size for previously answered questions
ENABLE_TRACEMALLOC = os.getenv("ENABLE_TRACEMALLOC", "0") == "1"

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tarunbot")

# ---------------- App state ----------------
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

# Simple LRU cache for answers: key=question (string), value=dict(answer_json)
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.od = OrderedDict()

    def get(self, key: str):
        if key in self.od:
            val = self.od.pop(key)
            self.od[key] = val
            return val
        return None

    def put(self, key: str, value):
        if key in self.od:
            self.od.pop(key)
        elif len(self.od) >= self.capacity:
            self.od.popitem(last=False)
        self.od[key] = value

answer_cache = LRUCache(CACHE_SIZE)

# ---------------- util: extract vector ----------------
def _extract_vector_from_embedding_obj(obj: Any) -> List[float]:
    if isinstance(obj, (list, tuple)):
        return list(obj)
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if hasattr(obj, "values"):
        vals = getattr(obj, "values")
        if isinstance(vals, (list, tuple)):
            return list(vals)
    if hasattr(obj, "embedding"):
        vals = getattr(obj, "embedding")
        if isinstance(vals, (list, tuple)):
            return list(vals)
    if isinstance(obj, dict):
        if "values" in obj:
            return list(obj["values"])
        if "embedding" in obj:
            return list(obj["embedding"])
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

# ---------------- GeminiEmbeddings (light) ----------------
class GeminiEmbeddings:
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
            batch = normalized[i:i + self.batch_size]
            vectors = self._embed_batch(batch)
            all_vectors.extend(vectors)
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

# ---------------- tracemalloc optional ----------------
def _maybe_start_tracemalloc():
    if ENABLE_TRACEMALLOC and not _services.get("tracemalloc_enabled", False):
        try:
            import tracemalloc as _tracemalloc
            _tracemalloc.start()
            _services["tracemalloc_enabled"] = True
            logger.info("tracemalloc started.")
        except Exception as e:
            logger.exception("Failed to start tracemalloc: %s", e)
            _services["tracemalloc_enabled"] = False

# ---------------- lazy init ----------------
def initialize_services():
    global _services_initialized
    if _services_initialized:
        return
    with _init_lock:
        if _services_initialized:
            return

        if ENABLE_TRACEMALLOC:
            _maybe_start_tracemalloc()

        # Gemini client
        try:
            if GEMINI_API_KEY:
                from google import genai
                gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                _services["gemini_client"] = gemini_client
                logger.info("Gemini client initialized.")
            else:
                _services["gemini_client"] = None
                logger.warning("GEMINI_API_KEY not set; gemini disabled.")
        except Exception as e:
            logger.exception("Failed to init Gemini client: %s", e)
            _services["gemini_client"] = None

        # Qdrant client
        lc_available = False
        if QDRANT_URL:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http import models as rest_models
                try:
                    from langchain_community.vectorstores import Qdrant as LCQdrant
                    lc_available = True
                except Exception:
                    lc_available = False
                qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
                _services["qdrant_client"] = qdrant_client
                logger.info("Qdrant client initialized.")
            except Exception as e:
                logger.exception("Failed to init Qdrant client: %s", e)
                _services["qdrant_client"] = None
                lc_available = False
        else:
            _services["qdrant_client"] = None

        # embeddings wrapper
        if _services["gemini_client"]:
            try:
                embeddings = GeminiEmbeddings(client=_services["gemini_client"], model=EMBED_MODEL)
                _services["embeddings"] = embeddings
                try:
                    sample_vec = embeddings.embed_query("test embedding dimension")
                    _services["embedding_dim"] = len(sample_vec)
                    logger.info("Detected embedding dim: %d", _services["embedding_dim"])
                except Exception:
                    _services["embedding_dim"] = None
            except Exception as e:
                logger.exception("Failed to create embeddings wrapper: %s", e)
                _services["embeddings"] = None
                _services["embedding_dim"] = None
        else:
            _services["embeddings"] = None
            _services["embedding_dim"] = None

        # Qdrant collection (non-destructive create if missing)
        if _services["qdrant_client"] and _services["embeddings"]:
            try:
                from qdrant_client.http import models as rest_models
                try:
                    _services["qdrant_client"].get_collection(collection_name=COLLECTION_NAME)
                    logger.info("Qdrant collection exists: %s", COLLECTION_NAME)
                except Exception:
                    dim = _services["embedding_dim"] or int(os.getenv("DEFAULT_EMBED_DIM", "1536"))
                    try:
                        _services["qdrant_client"].create_collection(
                            collection_name=COLLECTION_NAME,
                            vectors_config=rest_models.VectorParams(size=dim, distance=rest_models.Distance.COSINE),
                        )
                        logger.info("Created Qdrant collection '%s' dim=%d", COLLECTION_NAME, dim)
                    except Exception as e:
                        logger.exception("Failed to create qdrant collection: %s", e)
                # initialize langchain wrapper if available
                if lc_available:
                    try:
                        from langchain_community.vectorstores import Qdrant as LCQdrant
                        vectorstore = LCQdrant(client=_services["qdrant_client"], collection_name=COLLECTION_NAME, embeddings=_services["embeddings"])
                        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
                        _services["vectorstore"] = vectorstore
                        _services["retriever"] = retriever
                    except Exception as e:
                        logger.exception("Failed to init LangChain Qdrant wrapper: %s", e)
                        _services["vectorstore"] = None
                        _services["retriever"] = None
            except Exception as e:
                logger.exception("Qdrant collection handling error: %s", e)

        _services_initialized = True
        logger.info("Service initialization complete.")

# ---------------- Gemini LLM caller (blocking) ----------------
def ask_gemini_with_retry(prompt: str, retries: int = 2, delay: float = 1.0, thinking_budget: int = 10) -> str:
    """
    Blocking call; will be executed inside asyncio.to_thread in the async endpoint to avoid blocking the event loop.
    """
    initialize_services()
    client = _services.get("gemini_client")
    if not client:
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

# ---------------- FastAPI ----------------
app = FastAPI(title="TarunBot (render-guaranteed-response)")

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
    # Keep startup light: do NOT call initialize_services here to avoid heavy work on health checks.
    logger.info("TarunBot started. Services will initialize lazily on demand.")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "Hello from TarunBot (render-guaranteed-response)!"}

@app.get("/memory")
def memory(include_tracemalloc: bool = False):
    try:
        import psutil
    except Exception:
        return {"error": "psutil not installed; install with `pip install psutil`."}
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    rss = getattr(mem, "rss", None)
    vms = getattr(mem, "vms", None)
    resp = {
        "pid": proc.pid,
        "rss_bytes": rss,
        "rss_mb": round(rss / (1024 ** 2), 2) if rss else None,
        "vms_bytes": vms,
        "vms_mb": round(vms / (1024 ** 2), 2) if vms else None,
        "services_initialized": _services_initialized,
    }
    if include_tracemalloc and ENABLE_TRACEMALLOC and _services.get("tracemalloc_enabled", False):
        import tracemalloc
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")[:10]
        top = []
        for stat in top_stats:
            tb = stat.traceback
            frames = []
            for frame in tb[:3]:
                frames.append({"filename": frame.filename, "lineno": frame.lineno, "name": frame.name})
            top.append({"size_bytes": stat.size, "count": stat.count, "frames": frames})
        resp["tracemalloc_top"] = top
    return resp

# ---------------- fallback answer generator ----------------
def _safe_fallback_answer(question: str) -> dict:
    # Minimal, safe, non-prescriptive medical guidance
    return {
        "greeting": "Hello — thanks for your question.",
        "possible_causes": "I can't access the live assistant right now. Possible causes could include common, non-urgent issues such as mild viral infections, minor injuries, or temporary irritation — depending on symptoms.",
        "self_care": "If symptoms are mild, consider rest, hydration, and over-the-counter options like acetaminophen or ibuprofen for pain (follow product instructions). For topical issues, keep the area clean and avoid irritants.",
        "when_to_see_doctor": "Seek medical attention if you have severe pain, difficulty breathing, high fever, worsening symptoms, sudden neurological changes, or any other urgent red flags.",
        "closing": "Regards, AI Doctor Assistant (fallback due to high load)"
    }

# ---------------- /ask endpoint: guaranteed response ----------------
@app.post("/ask")
async def ask_bot(query: Query):
    """
    Attempts to acquire semaphore immediately (very short timeout).
    - If acquired: run normal flow (retriever -> LLM via to_thread), cache answer, return it.
    - If not acquired: try to return cached answer; if not cached, return safe fallback JSON immediately.
    """
    # Do lazy init only when necessary
    initialize_services()

    sem: asyncio.Semaphore = _services.get("semaphore", asyncio.Semaphore(MAX_CONCURRENT_REQUESTS))
    acquired = False
    try:
        # Immediate attempt to acquire; tiny timeout to avoid blocking
        try:
            await asyncio.wait_for(sem.acquire(), timeout=SEMAPHORE_WAIT_SECONDS)
            acquired = True
        except asyncio.TimeoutError:
            # Busy: return cached answer or fallback
            q_key = query.question.strip().lower()
            cached = answer_cache.get(q_key)
            if cached is not None:
                logger.info("Returning cached answer for question (busy): %s", q_key)
                # Mark that this is cached and busy
                cached_with_meta = dict(cached)
                cached_with_meta["_cached_fallback"] = True
                return {"answer": cached_with_meta}
            logger.info("No cached answer; returning safe fallback (busy).")
            fallback = _safe_fallback_answer(query.question)
            # include indicator so the client knows it was a fallback
            fallback["_fallback_due_to_load"] = True
            return {"answer": fallback}

        # We got a slot: perform normal retrieval + LLM call.
        retriever = _services.get("retriever")
        context = ""
        if retriever:
            try:
                # run retrieval in a thread to avoid blocking
                def do_retrieve(q: str):
                    try:
                        return retriever.get_relevant_documents(q)
                    except Exception:
                        try:
                            return retriever.retrieve(q)
                        except Exception:
                            return []
                docs = await asyncio.to_thread(do_retrieve, query.question)
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

        # Call the blocking LLM function in a thread
        raw_answer = await asyncio.to_thread(ask_gemini_with_retry, gemini_prompt)

        # Parse JSON if LLM returned JSON; otherwise place raw text into possible_causes
        try:
            answer_json = json.loads(raw_answer)
        except Exception:
            answer_json = {
                "greeting": "Hello!",
                "possible_causes": raw_answer,
                "self_care": "",
                "when_to_see_doctor": "",
                "closing": "Regards, AI Doctor Assistant",
            }

        # Cache the answer for future busy moments
        try:
            q_key = query.question.strip().lower()
            answer_cache.put(q_key, answer_json)
        except Exception:
            logger.exception("Failed to cache answer.")

        # Return normal answer
        return {"answer": answer_json}
    finally:
        if acquired:
            sem.release()
