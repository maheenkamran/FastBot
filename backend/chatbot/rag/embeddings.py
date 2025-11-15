from langchain_huggingface import HuggingFaceEmbeddings

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