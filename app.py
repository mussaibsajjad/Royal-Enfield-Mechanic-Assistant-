#!/usr/bin/env python3

import os
import logging
import requests
import openai
import streamlit as st
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# ─── Setup & env ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
load_dotenv()

# Azure Cognitive Search
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
SEARCH_INDEX    = os.getenv("AZURE_SEARCH_INDEX", "")
SEARCH_API_KEY  = os.getenv("AZURE_SEARCH_API_KEY", "")

# Azure OpenAI
openai.api_type    = "azure"
openai.api_base    = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")   # e.g. https://<your-resource>.openai.azure.com
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")            # e.g. "2023-05-15"
openai.api_key     = os.getenv("AZURE_OPENAI_API_KEY", "")
EMBED_MODEL        = os.getenv("AZURE_EMBED_DEPLOYMENT", "")              # your embedding deployment
CHAT_MODEL         = os.getenv("AZURE_CHAT_DEPLOYMENT", "")               # your gpt-4o deployment

# ─── Vector Search via REST ────────────────────────────────────
def vector_search(query: str, k: int = 7):
    # 1) Embed the query
    logging.info("⏳ Embedding query…")
    emb = openai.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding

    # 2) Call Azure Search REST
    url = f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX}/docs/search?api-version=2023-11-01"
    body = {
        "top": k,
        "vectorQueries": [
            {
                "fields": "contentVector",
                "vector": emb,
                "k": k,
                "kind": "vector",
                "exhaustive": False
            }
        ]
    }
    logging.info("⏳ Running vector search…")
    resp = requests.post(
        url,
        json=body,
        headers={
            "Content-Type": "application/json",
            "api-key": SEARCH_API_KEY
        }
    )
    resp.raise_for_status()
    hits = resp.json().get("value", [])
    logging.info(f"✅ Retrieved {len(hits)} chunks")
    # return list of (id, content)
    return [(h["id"], h["content"]) for h in hits]


# ─── Ask GPT-4o to answer over those chunks ─────────────────────
def answer_with_gpt4o(question: str, chunks: list[tuple[str, str]]):
    if not chunks:
        return "❗️Sorry, I couldn’t find any relevant documentation to answer that."

    # Log out exactly what chunks we're sending
    logging.info("🔍 Retrieved chunks:")
    for cid, text in chunks:
        clean_text = text[:100].replace('\n', ' ')
        logging.info(f" • {cid}: {clean_text}…")

    # build a system prompt
    system = {
        "role": "system",
        "content": "You are a certified Royal Enfield mechanic. Answer clearly and step-by-step."
    } 

    # assemble the context as inline citations
    context = "\n\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(chunks))
    user = {
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer in a concise, step-by-step format."
        )
    }

    logging.info("⏳ Generating answer via GPT-4o…")
    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[system, user],
        temperature=0.4,
        max_tokens=512
    )
    return resp.choices[0].message.content.strip()


# ─── Streamlit App ─────────────────────────────────────────────
st.set_page_config(
    page_title="Royal Enfield Mechanic Assistant",
    layout="wide",
)

st.title("🔧 Royal Enfield Mechanic Assistant")
st.sidebar.header("Settings")
k = st.sidebar.slider("Number of context chunks (k)", 1, 8, 4)

# initialize history
if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.text_input("Ask a question about your Royal Enfield…", key="question_input")
if st.button("🔍 Ask"):
    if prompt:
        with st.spinner("Thinking…"):
            # first fetch the top-k chunks
            hits = vector_search(prompt, k=k)
            # then ask GPT-4o to answer
            answer = answer_with_gpt4o(prompt, hits)
        st.session_state.history.append((prompt, answer))

st.markdown("## Conversation")
for q, a in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
