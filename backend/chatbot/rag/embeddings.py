from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model(model_name):
    """
    Returns a LangChain embedding model compatible with Chroma.
    """
    print(f"🔹 Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)
