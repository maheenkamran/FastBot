
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import numpy as np

def get_local_embeddings(model_name: str = None):
    """
    Returns a HuggingFaceEmbeddings instance that performs embeddings locally.
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def embed_texts(embeddings, texts: List[str]) -> np.ndarray:
    """
    Returns 2D numpy array of embeddings (n_texts, dim)
    """
    vecs = embeddings.embed_documents(texts)
    import numpy as np
    return np.array(vecs)


def get_embeddings_model(model_name):
    
    #Returns a LangChain embedding model compatible with Chroma.
    print(f"🔹 Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)



'''
Whoever calls get_embeddings_model(model_name) gets back a HuggingFaceEmbeddings object.
Because it is an object it can access methods of its class.

That object contains the loaded sentence-transformer model and provides special methods like:
.embed_query(text) → turns a user's question into an embedding vector.
.embed_documents(docs) → turns document text chunks into embeddings.

When your RAG calls get_embeddings_model(...), it receives a ready-to-use embedding model 
that can convert both questions and document text into vectors for similarity search.
'''