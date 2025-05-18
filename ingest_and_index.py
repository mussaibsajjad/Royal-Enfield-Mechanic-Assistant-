#!/usr/bin/env python3
import os
import sys
import io
import re
import logging
from dotenv import load_dotenv

# ─── Step 1: Imports & Environment Setup ──────────────────────────────────────
try:
    # Blob storage
    from azure.storage.blob import ContainerClient
    from azure.core.credentials import AzureNamedKeyCredential

    # Cognitive Search
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    # PDF & text splitting
    from PyPDF2 import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # OpenAI
    import openai

except ImportError as e:
    print(f"[✖] Missing required library: {e.name}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
load_dotenv()
logging.info("[✔] Imports loaded and .env variables read")

# ─── Read & validate core settings ───────────────────────────────────────────
SERVICE_ENDPOINT    = os.getenv("AZURE_SEARCH_ENDPOINT")
INDEX_NAME          = os.getenv("AZURE_SEARCH_INDEX")
API_KEY             = os.getenv("AZURE_SEARCH_API_KEY")
BLOB_ACCOUNT_URL    = os.getenv("BLOB_ACCOUNT_URL")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
BLOB_ACCOUNT_KEY    = os.getenv("BLOB_ACCOUNT_KEY")
EMBED_MODEL         = os.getenv("AZURE_EMBED_DEPLOYMENT")

for name, val in {
    "AZURE_SEARCH_ENDPOINT":  SERVICE_ENDPOINT,
    "AZURE_SEARCH_INDEX":     INDEX_NAME,
    "AZURE_SEARCH_API_KEY":   API_KEY,
    "BLOB_ACCOUNT_URL":       BLOB_ACCOUNT_URL,
    "BLOB_CONTAINER_NAME":    BLOB_CONTAINER_NAME,
    "BLOB_ACCOUNT_KEY":       BLOB_ACCOUNT_KEY,
    "AZURE_EMBED_DEPLOYMENT": EMBED_MODEL,
}.items():
    if not val:
        logging.error(f"Missing required .env variable: {name}")
        sys.exit(1)

logging.info(f"Azure Search → {SERVICE_ENDPOINT}/{INDEX_NAME}")
logging.info(f"Blob Storage → {BLOB_ACCOUNT_URL} (container: {BLOB_CONTAINER_NAME})")

# ─── Step 2: Connect to Azure Cognitive Search Index ─────────────────────────
def connect_to_search_index():
    creds  = AzureKeyCredential(API_KEY)
    client = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=creds)
    logging.info("✅ Connected to Azure Cognitive Search index")
    return client

# ─── Step 3: Fetch PDFs from Blob Storage and Extract Text ───────────────────
def fetch_and_extract_pdfs():
    account_name = BLOB_ACCOUNT_URL.split("//")[1].split(".")[0]
    cred = AzureNamedKeyCredential(account_name, BLOB_ACCOUNT_KEY)

    container = ContainerClient(
        account_url=BLOB_ACCOUNT_URL,
        container_name=BLOB_CONTAINER_NAME,
        credential=cred
    )
    logging.info(f"✅ Connected to Blob container '{BLOB_CONTAINER_NAME}'")

    docs = []
    for blob in container.list_blobs():
        if not blob.name.lower().endswith(".pdf"):
            continue

        logging.info(f"Downloading blob: {blob.name}")
        data = container.download_blob(blob).readall()

        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages)

        docs.append({"id": blob.name, "text": full_text})
        logging.info(f"Extracted {len(pages)} pages from {blob.name}")

    logging.info(f"🎉 Total PDFs downloaded & extracted: {len(docs)}")
    return docs

# ─── Step 4: Chunk text & index embeddings ────────────────────────────────────
def chunk_and_index(docs, search_client):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Configure Azure OpenAI for the new SDK
    openai.api_type    = "azure"
    openai.api_base    = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    openai.api_key     = os.getenv("AZURE_OPENAI_API_KEY")

    batch = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        logging.info(f"Splitting '{doc['id']}' into {len(chunks)} chunks")

        base = os.path.splitext(doc["id"])[0]  # drop “.pdf”
        for i, chunk in enumerate(chunks):
            # sanitize chunk ID to only [A-Za-z0-9_-]
            raw_id  = f"{base}_chunk_{i}"
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", raw_id)

            # get embedding
            resp   = openai.embeddings.create(model=EMBED_MODEL, input=[chunk])
            vector = resp.data[0].embedding

            batch.append({
                "id":            safe_id,
                "content":       chunk,
                "contentVector": vector
            })

    result = search_client.upload_documents(documents=batch)
    logging.info(f"✅ Indexed {len(batch)} total chunks into Azure Search")
    return result

# ─── Main Flow through Step 4 ─────────────────────────────────────────────────
if __name__ == "__main__":
    client = connect_to_search_index()
    docs   = fetch_and_extract_pdfs()
    chunk_and_index(docs, client)
