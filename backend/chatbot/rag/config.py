''' A config file (short for configuration file) is a file used to store settings or constants that your program 
needs — things that don't usually change during execution.
So instead of hardcoding these values in multiple places, you keep them in one file — 
making your code cleaner, reusable, and easier to update. '''

import os
#provides functions to interact with the operating system.
#Used here for working with file paths 

# Get the absolute path to the project root (the main FastBot folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(os.path.abspath(__file__))))))
# os.path.abspath(__file__) gives full path of file
# 4 times step to root folder 

DATA_PATH = os.path.join(BASE_DIR, "docs", "QA.pdf")  #joins path via \ like docs\QA.pdf
           
VECTOR_DB_PATH = os.path.join(BASE_DIR, "backend", "chatbot", "rag", "chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#constants(caps+_) string storing name of our model

QA_MODEL = "deepset/roberta-base-squad2"
