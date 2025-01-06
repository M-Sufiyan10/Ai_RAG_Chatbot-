import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as vector_pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
import os
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone
import uuid


#loading necessary keys
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_HOST")
embeddings = GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/embedding-001")
model = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
index_name = "rag"


#returns the loaded document
def load_documents(docs,type='pdf'):
    if type =='pdf':
        loader=PyPDFLoader(docs)
        docs=loader.load()
    else:
        loader=WebBaseLoader(docs)
        docs=loader.load()
    return docs

#returns the chunking of loaded documents
def chunking(docs,chunk_size=1000,chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n",".\n",".\n\n"," ",""]
    )
    texts = text_splitter.split_documents(docs)
    return texts


#it will create chain of vector retriever and memory
def process(texts,embeddings,index_name,model):
    vectordb = vector_pinecone.from_documents(documents=texts, embedding=embeddings, index_name=index_name)
    retriever = vectordb.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever, memory=memory)
    return chain


#it will return the response of the query
def get_query(chain,query):
    response = chain({"question": query})
    return response["answer"]

st.title("Language Chain App")
option=st.selectbox('choose document type',['Web URL','PDF'])
if option=='Web URL':
    st.write("Enter a URL to process queries with RAG.")
    url = st.text_input("Enter URL:")        
    docs = load_documents(url,option)
    texts=chunking(docs)
    chain=process(texts,embeddings,index_name,model)
    query = st.text_input("Enter your query:")
    if query:
        try:
            st.write("### Response")
            response=get_query(chain,query)
            st.write(response)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.write("Upload multiple PDFs to process queries with RAG.")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Process all uploaded files
        all_texts = []
        for index,uploaded_file in enumerate(uploaded_files):
            # Save each file temporarily
            temp_file = f"./temp_{uuid.uuid4().hex}.pdf"
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())

            # Load and process the PDF
            docs = load_documents(temp_file)

            # Split documents
            texts = chunking(docs)
            all_texts.extend(texts)  
        # Create vector store with all texts
        chain=process(all_texts,embeddings,index_name,model)
        # # User input query
        query = st.text_input("Enter your query:")
        if query:
            try:
                response=get_query(chain,query)
                st.write("### Response")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
