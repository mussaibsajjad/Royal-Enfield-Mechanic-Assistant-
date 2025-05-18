#!/usr/bin/env python3
import os
import json
import logging
import requests
import openai
from dotenv import load_dotenv

# ─── Step 0: Env & Logging ──────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── Configuration ──────────────────────────────────────────────
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")           # e.g. https://<your-search>.search.windows.net
SEARCH_INDEX    = os.getenv("AZURE_SEARCH_INDEX")              # e.g. azureblob-index
SEARCH_KEY      = os.getenv("AZURE_SEARCH_API_KEY")

openai.api_type    = "azure"
openai.api_base    = os.getenv("AZURE_OPENAI_ENDPOINT")        # e.g. https://<your-openai>.openai.azure.com
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")     # e.g. 2023-05-15
openai.api_key     = os.getenv("AZURE_OPENAI_API_KEY")
EMBED_MODEL       = os.getenv("AZURE_EMBED_DEPLOYMENT")        # e.g. ada-002-emb

# ─── Vector‐powered search via REST ─────────────────────────────
def vector_search(query: str, k: int = 5):
    # 1) embed the query
    logging.info("Generating embedding via Azure OpenAI…")
    resp = openai.embeddings.create(model=EMBED_MODEL, input=[query])
    vector = resp.data[0].embedding

    # 2) build REST payload
    payload = {
        "search": "*",
        "select": "id,content",
        "vectorQueries": [
            {
                "vector": vector,
                "fields": "contentVector",
                "k": k,
                "kind": "vector"
            }
        ]
    }

    url = f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX}/docs/search?api-version=2023-11-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": SEARCH_KEY
    }

    logging.info("POSTing vector query to Azure Cognitive Search…")
    logging.debug(json.dumps(payload, indent=2))

    r = requests.post(url, headers=headers, json=payload)
    logging.info(f"Response status: {r.status_code}")
    if not r.ok:
        logging.error("Error body:\n" + r.text)
        r.raise_for_status()

    hits = r.json().get("value", [])
    logging.info(f"🔎 Top {len(hits)} results for “{query}”:")
    for h in hits:
        print(f"- {h['id']}:")
        snippet = h["content"].replace("\n", " ")[:200]
        print(f"    {snippet}…\n")

if __name__ == "__main__":
    test_query = "How do I change the oil?"
    vector_search(test_query, k=5)
