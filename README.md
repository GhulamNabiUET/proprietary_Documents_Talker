# 🧾 Talk with Proprietary Files

A Streamlit-based application that allows users to **upload PDF documents** and interactively **ask questions** about their contents using LLM-powered retrieval-based QA.

This app uses:
- LangChain for document loading and QA chaining
- FAISS for vector similarity search
- Hugging Face transformers for language modeling
- Streamlit for the user interface

---

## 🚀 Features

- 📄 Upload multiple PDF files
- 📚 Extract and chunk documents into text
- 🤖 Embed text using `sentence-transformers/all-MiniLM-L6-v2`
- 🧠 Ask questions and receive LLM-generated answers based on uploaded content
- 🔍 Uses vector similarity search with FAISS
- 🧵 Keeps history of Q&A and source context per session

---

## 📦 Requirements

Install dependencies via pip:

```bash
pip install streamlit langchain faiss-cpu sentence-transformers transformers
