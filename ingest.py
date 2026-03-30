import os
import time

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

pdfs_file = "data/" # path to PDFs folder
urls_file = "urls.txt" # txt file with one URL per line

def load_pdfs(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    docs=loader.load()
    print("Length of pdf pages-", len(docs))
    return docs

def load_urls(urls_file):
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    docs = []

    for url in urls:
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                requests_kwargs={
                    "headers": {"User-Agent": "Mozilla/5.0"},
                    "timeout": 20
                }
            )
            data = loader.load()
            if data:
                docs.extend(data)
                print(f"Loaded-{url}")
            else:
                print(f"No data-{url}")

            time.sleep(2)  # avoid blocking

        except Exception as e:
            print(f"Skipped: {url}")

    print("Total pages loaded-", len(docs))
    return docs

# Initialize RecursiveCharacterTextSplitter
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
pdf_docs = load_pdfs(pdfs_file)
web_docs = load_urls(urls_file)
all_docs = pdf_docs + web_docs
all_docs = [d for d in all_docs if d.page_content.strip()]

chunks = create_chunks(all_docs)
print("Total chunks-", len(chunks))

embedding_model = get_embedding_model()

DB_FAISS_PATH="vectorstore/db_faiss"
os.makedirs(DB_FAISS_PATH, exist_ok=True)

db = FAISS.from_documents(
    chunks,
    embedding_model,
    normalize_L2=True
)



