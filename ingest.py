# This file is used to break text into vectors for Pinecone 

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# look for .env and loads apis 
load_dotenv()

def run_ingestion(file_name):
    print(f"Starting to read: {file_name}...")

    # extract raw text from the pdf 
    loader = PyPDFLoader(f"data/{file_name}")
    raw_documents = loader.load()

    # we take chuncks of 1000 words and transform them 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(raw_documents)
    print(f"Created {len(docs)} small chunks of knowledge.")

    # gemini does the math and gives us set of number that represents the chunck 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # store our vectors in the pinecone database 
    index_name = "company-expert"
    
    PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    
    print("Success! Your Pinecone 'library' is now full of knowledge.")

if __name__ == "__main__":
    # Change 'test.pdf' to whatever your file is called!
    run_ingestion("company_info.pdf")