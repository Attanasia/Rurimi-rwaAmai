from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from shona_data import shona_dict   

print(f"Loading {len(shona_dict)} translation pairs...")

#Convert each translation pair into a Document
documents = []
for english, shona in shona_dict.items():
   
    content = f"English phrase: {english} | Shona translation: {shona}"
    metadata = {
        "english": english,
        "shona": shona,
        "category": "translation"
    }
    doc = Document(page_content=content, metadata=metadata)
    documents.append(doc)

print(f"Created {len(documents)} documents")


print("Loading embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

#Create and persist the vector store
print("Creating Chroma vector store...")
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./shona_vector_db"  
)

# Persist the database
vector_store.persist()
print("Vector store saved to './shona_vector_db'")

# Test retrieval
print("\n--- Testing retrieval ---")
test_query = "How do you say good morning?"
results = vector_store.similarity_search(test_query, k=3)
print(f"Query: {test_query}")
print("Top matches:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")