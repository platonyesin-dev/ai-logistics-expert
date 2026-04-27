import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup & Security
load_dotenv()

st.set_page_config(page_title="Atlas Prime Expert", page_icon="🚛")
st.title("🚛 Atlas Prime Logistics Expert")
st.markdown("Ask me anything about our plans, regions, or safety protocols.")

#2.  We will use gemini again to transform prompt to the vector format
# We use the exact same embedding model as we did in ingest.py
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
index_name = "company-expert"
# Connect to the database we already filled
vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)

# This function decides which data to use gmail one or pdf guide 
def priority_retriever(query):
    # 1. Try to find info in the PDF first
    # We use a filter to look ONLY at PDF chunks
    pdf_results = vectorstore.similarity_search(
        query, 
        k=3, 
        filter={"source": "company_info.pdf"} 
    )
    
    # 2. If we found something in the PDF, return it immediately
    if pdf_results:
        st.toast("🔍 Logic: Match found in PDF manual.")
        print("Found answer in PDF.")
        return pdf_results
    
    # 3. If PDF is empty, try the Gmail data
    st.toast("📧 Logic: PDF silent. Searching Gmail archives...")
    print("PDF silent. Checking Gmail archives...")
    gmail_results = vectorstore.similarity_search(
        query, 
        k=3, 
        filter={"source": "gmail"}
    )
    return gmail_results

# 3. Initialize Groq as our chat brain 
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", # Updated name for 2026
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# 4. Create the "Chain" 
# Define a prompt that tells the AI how to use the retrieved info
system_prompt = (
    "You are a Senior Logistics Consultant for Atlas Prime. "
    "You have two sources of knowledge: \n"
    "1. THE MANUAL (Official rules/plans)\n"
    "2. CLIENT EXPERIENCE (Recent emails and situations)\n\n"
    "INSTRUCTIONS:\n"
    "- Always start with the official rule from the Manual.\n"
    "- THEN, check the Client Experience. If you find a relevant real-world situation, "
    "explain how it was handled (e.g., 'In a similar case last month, we did...').\n"
    "- If Gmail data shows a newer update than the PDF, prioritize the Gmail data but "
    "mention it is a recent change.\n\n"
    "CONTEXT:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# This chain handles combining the PDF snippets into one prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

# This is the final chain that links Pinecone (retriever) to the prompt logic
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

# Chat loop hisiry 
if "messages" not in st.session_state:
    # here we store our messages from the chat 
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting company manual..."):
            # This is where the magic happens: 
            # 1. Pinecone finds info. 2. Groq reads it. 3. Groq answers.
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"] # Note: it is now 'answer', not 'result'
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})