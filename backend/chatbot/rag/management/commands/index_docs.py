from django.core.management.base import BaseCommand
from backend.chatbot.rag.config import DATA_DIR, CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from backend.chatbot.rag.data_loader import load_pdfs_as_page_docs
from backend.chatbot.rag.embeddings import get_local_embeddings
from backend.chatbot.rag.retriever import build_or_load_chroma

class Command(BaseCommand):
    help = "Index PDFs into Chroma vector store"

    def handle(self, *args, **options):
        print("Loading pages from PDFs...")
        docs = load_pdfs_as_page_docs(str(DATA_DIR))
        if not docs:
            print("No documents found in", DATA_DIR)
            return
        print(f"Loaded {len(docs)} pages. Building vectorstore...")
        embeddings = get_local_embeddings(model_name=EMBEDDING_MODEL)
        db = build_or_load_chroma(docs, str(CHROMA_PERSIST_DIR), embeddings)
        print("Indexing completed and persisted at", CHROMA_PERSIST_DIR)
