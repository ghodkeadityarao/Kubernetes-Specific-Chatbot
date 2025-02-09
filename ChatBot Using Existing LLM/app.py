import os

import google.generativeai as genai
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

INDEX_PATH = "faiss_index"


def scrape_kubernetes_docs():
    """Scrapes Kubernetes documentation and returns the text."""
    base_url = "https://kubernetes.io/docs/"
    pages = [
        "concepts/overview/",
        "concepts/workloads/",
        "concepts/services-networking/",
        "concepts/configuration/",
        "concepts/storage/",
        "concepts/security/",
        "concepts/architecture/",
        "concepts/extend-kubernetes/",
        "concepts/cluster-administration/",
        "concepts/windows/",
        "tasks/",
    ]

    all_text = ""
    for page in pages:
        url = base_url + page
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator="\n", strip=True)
            all_text += f"\n\nPage: {url}\n" + page_text
        else:
            print(f"Failed to scrape {url}")
    
    return all_text


def get_text_chunks(text):
    """Splits the scraped text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def create_and_save_vector_store(text_chunks):
    """Creates FAISS vector store and saves it."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(INDEX_PATH)


def load_vector_store():
    """Loads the FAISS vector store if it exists."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None


def get_conversational_chain():
    """Returns the Gemini-based QA chain."""
    prompt_template = """
    You are an expert in Kubernetes. Answer the question in as much detail as possible using the provided context. 
    If the answer is not available in the context, first explain general knowledge about the topic and then state that 
    additional details are not available in the provided documentation. If the question is not relatedto kubernetes then
    reply that you can only answer kubernetes related queries.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question, vector_store):
    """Processes the user question using FAISS and Gemini."""
    docs = vector_store.similarity_search(user_question, k=5)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config("Chat with Kubernetes Docs")
    st.header("Chat with Kubernetes Documentation")

    vector_store = load_vector_store()

    if vector_store is None:
        st.write("Scraping Kubernetes Docs and creating FAISS index (only runs once)...")
        raw_text = scrape_kubernetes_docs()
        text_chunks = get_text_chunks(raw_text)
        create_and_save_vector_store(text_chunks)
        vector_store = load_vector_store()  

    user_question = st.text_input("Ask a question about Kubernetes:")

    if user_question and vector_store:
        user_input(user_question, vector_store)


if __name__ == "__main__":
    main()
