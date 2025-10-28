from langchain_community.vectorstores import Chroma

def create_or_load_db(splits, embedding_model, persist_dir):
    print("🔹 Creating/Loading vector database...")
    vectordb = Chroma.from_documents(splits, embedding_model, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb
