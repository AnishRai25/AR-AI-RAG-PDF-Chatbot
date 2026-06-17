# AR AI - RAG PDF Chatbot

## Overview

AR AI is an advanced Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions in natural language.

The application uses LangChain, FAISS, and Large Language Models to retrieve relevant information from documents and generate accurate responses with source citations.

## Features

* PDF Upload & Processing
* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Source Citations
* OpenAI Integration
* Ollama Local LLM Support
* FAISS Vector Database
* Streamlit User Interface

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* OpenAI
* Ollama
* PyPDF

## Architecture

PDF Upload
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
FAISS Vector Store
↓
Retriever
↓
LLM
↓
Answer + Citations

## Installation

pip install -r requirements.txt

streamlit run app.py

## Author

Anish Rai
BE Computer Engineering
