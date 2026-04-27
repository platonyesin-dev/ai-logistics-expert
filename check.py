import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Lets check if data uploaded to pinecone is actully legit 
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
index_name = "company-expert"

# Connect to the existing index in pinecone
vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Ask a test question
query = "What are the restrictions on jewelry?"
docs = vectorstore.similarity_search(query, k=1)

print("\n--- TEST SEARCH RESULT ---")
if docs:
    print(docs[0].page_content)
else:
    print("No data found!")