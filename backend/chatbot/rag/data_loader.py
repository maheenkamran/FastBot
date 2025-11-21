from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List
import os

def load_pdfs_as_page_docs(data_dir: str) -> List[Document]:
    """
    Load PDFs from data_dir and return a list of Document objects (one per page)
    with metadata {'source': filename, 'page': page_number}
    """
    data_path = Path(data_dir)
    docs = []
    for pdf_path in sorted(data_path.glob("*.pdf")):
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load_and_split()  # should give per-page docs in many versions
            # ensure metadata includes page numbers
            for i, p in enumerate(pages):
                meta = dict(p.metadata) if p.metadata else {}
                meta.update({"source": pdf_path.name, "page": i + 1})
                docs.append(Document(page_content=p.page_content, metadata=meta))
            print(f"Loaded {len(pages)} pages from {pdf_path.name}")
        except Exception as e:
            print(f"Failed to load {pdf_path.name}: {e}")
    return docs
