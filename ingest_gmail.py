import os.path
import base64
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
load_dotenv()

# we ask google only to read email nothing else
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = None # this variable will eventually hold digital key
    # token.json stores google login info for the fast login
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # check if creds are valid
    if not creds or not creds.valid:
        # if creds are not good asks google to refresh them 
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else: # first time login in case we have never entered google yet
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # once we get fresh keys we immediately write them in token.json
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def ingest_recent_emails():
    service = get_gmail_service()
    # Asks Google for a list of mails and specific charachteristics 
    results = service.users().messages().list(userId='me', maxResults=15, q='label:inbox').execute()
    # 'messages' holds a list of ids 
    messages = results.get('messages', [])

    docs = []
    print(f"Reading {len(messages)} emails...")
    
    # loop through emails to actually get basic content anout them 
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        # snippet takes few senteces from the gmail 
        snippet = msg_data.get('snippet', '')
        
        # We tag this as "gmail" so in the future computer knows difference 
        docs.append(Document(
            page_content=snippet,
            metadata={"source": "gmail", "id": msg['id']}
        ))

    # Save to Pinecone
    # takes every email nippet and tunrs it into vector set 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # after saves it in db pinecone
    PineconeVectorStore.from_documents(docs, embeddings, index_name="company-expert")
    print("Success! Emails added to your expert's experience.")

if __name__ == "__main__":
    ingest_recent_emails()