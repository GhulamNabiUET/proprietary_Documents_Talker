import streamlit as st
import numpy as np
import tempfile
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from typing import List

# Initialize Streamlit session state for question history and source documents
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'source' not in st.session_state:
    st.session_state['source'] = []

# Utility functions
def file_log(logentry):
    with open("file_ingest.log", "a") as file1:
        file1.write(logentry + "\n")
    print(logentry + "\n")

def load_single_document(file_path: str) -> Document:
    try:
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError("Unsupported document type")
        return loader.load()[0]
    except Exception as ex:
        file_log(f"{file_path} loading error: \n{ex}")
        return None

def load_document_batch(filepaths):
    logging.info("Loading document batch")
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in as_completed(futures)]
        return data_list

# Streamlit application
st.title("Talk with Proprietary Files")

# File upload section
uploaded_files = st.file_uploader("Upload your documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    temp_file_paths = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            temp_file_paths.append(temp_file_path)

    # Load and process documents
    documents = load_document_batch(temp_file_paths)

    st.write("Uploaded and processed documents:")
    for doc in documents[:1]:  # Displaying the first document for brevity
        st.write(doc.page_content)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)

    # Embedding using Huggingface
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # VectorStore Creation
    vectorstore = FAISS.from_documents(final_documents[:120], huggingface_embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    prompt_template = """
    Use the following piece of context to answer the question asked.
    Please try to provide the answer only based on the context and in some descriptive way as well

    {context}
    Question:{question}

    Helpful Answers:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retrievalQA = RetrievalQA.from_chain_type(
        llm=hf,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # Query input section
    query = st.text_input("💢:Enter your query:")

    if st.button("Get Answer"):
        with st.spinner('Processing...'):
            # Call the QA chain with the user query
            result = retrievalQA.invoke({"query": query})
            answer = result['result']
            source_documents = result['source_documents']

            # Store query and answer in session state history
            st.session_state['history'].append({"query": query, "answer": answer})
            st.session_state['source'].append({"question": query, "answer": answer, "documents": source_documents})

        st.write("Answer:")
        st.write(f"🚀 {answer}")

    # Display question history
    if st.session_state['history']:
        st.write("Question History")
        for i, entry in enumerate(st.session_state['history']):
            with st.expander(f"Query {i+1}: 🔴 {entry['query']}"):
                st.write(f"🚀 Answer: {entry['answer']}")

    # Display source documents
    if st.session_state['source']:
        st.write("Source Documents")
        for i, entry in enumerate(st.session_state['source']):
            with st.expander(f"Query {i+1}: 🔴 {entry['question']}"):
                st.write(f"🟡 Answer: {entry['answer']}")
                for doc in entry['documents']:
                    st.write(doc.page_content)

else:
    st.write("Please upload PDF documents to start.")
