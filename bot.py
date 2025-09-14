import streamlit as st
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import time
import threading

# ---------------- Load environment variables ----------------
load_dotenv()
QDRANT_URL = "https://77515254-76b4-47f6-9d4c-cf5417a3daba.eu-central-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "tarun_knowledge"

# ---------------- Qdrant setup ----------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ---------------- Gemini setup ----------------
from google import genai
from google.genai import types
from google.genai.errors import ServerError

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini_with_retry(prompt: str, retries=3, delay=2) -> str:
    """Send prompt to Gemini API with retry logic for 503 errors."""
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
        except ServerError as e:
            print(f"[Gemini attempt {attempt+1}] ServerError: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                return "Sorry, the model is currently overloaded. Please try again later."

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Tarun's Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.user {background-color: #DCF8C6; padding: 10px; border-radius: 15px; margin: 5px; text-align: right; width: fit-content; max-width: 70%; color:black;}
.bot {background-color: #E6E6E6; padding: 10px; border-radius: 15px; margin: 5px; text-align: left; width: fit-content; max-width: 70%; color:black;}
.typing {background-color: #E6E6E6; padding: 10px; border-radius: 15px; margin: 5px; width: fit-content; max-width: 50px; text-align: left; font-weight: bold; color: #555;}
.chat-container {display: flex; flex-direction: column;}
.user-row {display: flex; justify-content: flex-end;}
.bot-row {display: flex; justify-content: flex-start;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Ask anything about Tarun")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm Tarun's chatbot. Ask me anything about him."}
    ]

def render_messages():
    for chat in st.session_state["messages"]:
        if chat["role"] == "user":
            st.markdown(f'<div class="chat-container"><div class="user-row"><div class="user">{chat["content"]}</div></div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-container"><div class="bot-row"><div class="bot">{chat["content"]}</div></div></div>', unsafe_allow_html=True)

render_messages()

prompt = st.chat_input("Type your question here...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    render_messages()

    typing_placeholder = st.empty()
    stop_animation = False

    def typing_animation():
        i = 0
        while not stop_animation:
            dots = "." * ((i % 3) + 1)
            typing_placeholder.markdown(f'<div class="chat-container"><div class="bot-row"><div class="typing">{dots}</div></div></div>', unsafe_allow_html=True)
            time.sleep(0.5)
            i += 1

    t = threading.Thread(target=typing_animation)
    t.start()

    docs = retriever.get_relevant_documents(prompt)
    if docs:
        context = "\n".join([d.page_content for d in docs])
        gemini_prompt = f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {prompt}"
    else:
        gemini_prompt = prompt

    response = ask_gemini_with_retry(gemini_prompt)

    stop_animation = True
    t.join()
    typing_placeholder.empty()

    st.session_state["messages"].append({"role": "assistant", "content": response})
    render_messages()
