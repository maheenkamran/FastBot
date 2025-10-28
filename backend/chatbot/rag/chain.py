from transformers import pipeline
from .data_loader import load_and_split_docs
from .embeddings import get_embeddings_model
from .retriever import create_or_load_db
from .config import DATA_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL, QA_MODEL

def build_rag():
    splits = load_and_split_docs(DATA_PATH)
    embedding_model = get_embeddings_model(EMBEDDING_MODEL)
    vectordb = create_or_load_db(splits, embedding_model, VECTOR_DB_PATH)
    return vectordb, pipeline("question-answering", model=QA_MODEL)

def ask_question(query):
    vectordb, qa_pipeline = build_rag()
    docs = vectordb.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])
    result = qa_pipeline(question=query, context=context)
    return result["answer"]
