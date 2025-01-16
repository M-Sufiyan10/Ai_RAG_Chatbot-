# import streamlit as st
# from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Pinecone as vector_pinecone
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
# import os
# from dotenv import load_dotenv
# import pinecone
# from pinecone import Pinecone
# import uuid

# def configure_temperature():
#     """
#     Creates a temperature slider in the Streamlit sidebar for configuring the LLM.
#     Returns:
#         float: The selected temperature value.
#     """
#     st.sidebar.subheader("Temperature")
#     return st.sidebar.slider(
#         "Control the randomness of responses",
#         min_value=0.0,
#         max_value=1.0,
#         value=0.5,
#         step=0.1,
#         help="Lower values make the model more deterministic, higher values make it more creative and randomness."
#     )

# def configure_max_tokens():
#     """
#     Creates a max tokens slider in the Streamlit sidebar for configuring the LLM.
#     Returns:
#         int: The selected maximum token value.
#     """
#     st.sidebar.subheader("Max Tokens")
#     return st.sidebar.slider(
#         "Limit the response length",
#         min_value=50,
#         max_value=500,
#         value=200,
#         step=10,
#         help="Defines the maximum number of tokens (words/pieces) in the response."
#     )


# #loading necessary keys
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# pinecone_environment = os.getenv("PINECONE_HOST")
# embeddings = GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/embedding-001")
# model = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key,max_tokens=configure_max_tokens(),temperature= configure_temperature())
# index_name = "rag"


# #returns the loaded document
# def load_documents(docs,type='pdf'):
#     if type =='pdf':
#         loader=PyPDFLoader(docs)
#         docs=loader.load()
#     else:
#         loader=WebBaseLoader(docs)
#         docs=loader.load()
#     return docs

# #returns the chunking of loaded documents
# def chunking(docs,chunk_size=1000,chunk_overlap=100):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         add_start_index=True,
#         separators=["\n\n",".\n",".\n\n"," ",""]
#     )
#     texts = text_splitter.split_documents(docs)
#     return texts


# #it will create chain of vector retriever and memory
# def process(texts,embeddings,index_name,model):
#     vectordb = vector_pinecone.from_documents(documents=texts, embedding=embeddings, index_name=index_name)
#     retriever = vectordb.as_retriever()
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     chain = ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever, memory=memory)
#     return chain


# #it will return the response of the query
# def get_query(chain,query):
#     response = chain({"question": query})
#     return response["answer"]

# st.title("Language Chain App")
# option=st.selectbox('choose document type',['WEB','PDF'])
# if option == 'WEB':
#     st.write("Enter a URL")
#     url = st.text_input("Enter URL:")
#     if url:
#         if not url.startswith("http://") and not url.startswith("https://"):
#             st.error("Invalid URL. Please include 'http://' or 'https://'")
#         else:
#             try:
#                 docs = load_documents(url, option)
#                 texts = chunking(docs)
#                 chain = process(texts, embeddings, index_name, model)

#                 if 'messages' not in st.session_state:
#                     st.session_state.messages = []
#                 for message in st.session_state.messages:
#                     st.chat_message(message['role']).markdown(message['content'])    
#             except Exception as e:
#                 st.error(f"Error loading the document: {e}")
#     else:
#         st.warning("Please enter a valid URL.")
# else:
#     st.write("Upload multiple PDFs to process queries with RAG.")
#     uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
#     if uploaded_files:
#         all_texts = []
#         for index,uploaded_file in enumerate(uploaded_files):
#             temp_file = f"./temp_{uuid.uuid4().hex}.pdf"
#             with open(temp_file, "wb") as file:
#                 file.write(uploaded_file.getvalue())
#             docs = load_documents(temp_file)
#             texts = chunking(docs)
#             all_texts.extend(texts)  
#         chain=process(all_texts,embeddings,index_name,model)


# if 'messages' not in st.session_state:
#     st.session_state.messages=[]
# for message in st.session_state.messages:
#     st.chat_message(message['role']).markdown(message['content'])
# query = st.chat_input("Enter your query:")
# if query:
#     try:
#         st.chat_message('user').markdown(query)
#         response = get_query(chain, query)
#         st.session_state.messages.append({'role': 'user', 'content': query})
#         st.chat_message('assistant').markdown(response)
#         st.session_state.messages.append({'role': 'assistant', 'content': response})
#     except Exception as e:
#         st.error(f"An error occurred: {e}")



import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as vector_pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
import os
from dotenv import load_dotenv
import pinecone
import uuid

# Sidebar configurations
def configure_temperature():
    st.sidebar.subheader("Temperature")
    return st.sidebar.slider("Control the randomness of responses", 0.0, 1.0, 0.5, 0.1)

def configure_max_tokens():
    st.sidebar.subheader("Max Tokens")
    return st.sidebar.slider("Limit the response length", 50, 500, 200, 10)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_HOST")

# Initialize embeddings and model
embeddings = GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/embedding-001")
model = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key,
    max_tokens=configure_max_tokens(),
    temperature=configure_temperature()
)
index_name = "rag"

def load_documents(docs, doc_type='pdf'):
    loader = PyPDFLoader(docs) if doc_type == 'pdf' else WebBaseLoader(docs)
    return loader.load()

def chunking(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def process(texts):
    vectordb = vector_pinecone.from_documents(texts, embedding=embeddings, index_name=index_name)
    retriever = vectordb.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever, memory=memory)

def get_query(chain, query):
    response = chain.invoke({"question": query})  # Replaced deprecated __call__
    return response["answer"]

# Streamlit UI
st.title("Language Chain App")
option = st.selectbox('Choose document type', ['WEB', 'PDF'])

if option == 'WEB':
    url = st.text_input("Enter URL:")
    if url:
        if url.startswith(("http://", "https://")):
            try:
                docs = load_documents(url, 'web')
                texts = chunking(docs)
                chain = process(texts)
            except Exception as e:
                st.error(f"Error loading the document: {e}")
        else:
            st.error("Invalid URL. Please include 'http://' or 'https://'")
else:
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        all_texts = []
        for file in uploaded_files:
            temp_path = f"./temp_{uuid.uuid4().hex}.pdf"
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            docs = load_documents(temp_path)
            all_texts.extend(chunking(docs))
        chain = process(all_texts)

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg['role']).markdown(msg['content'])

query = st.chat_input("Enter your query:")
if query:
    try:
        st.chat_message('user').markdown(query)
        response = get_query(chain, query)
        st.session_state.messages.append({'role': 'user', 'content': query})
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
    except Exception as e:
        st.error(f"An error occurred: {e}")
