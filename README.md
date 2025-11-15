AmbedkarGPT-Intern-Task

This project is a submission for the Kalpit Pvt Ltd AI Intern Assignment. It is a simple, command-line Q&A system that implements a complete Retrieval-Augmented Generation (RAG) pipeline using LangChain, Ollama, and ChromaDB.

The system ingests a provided text file (speech.txt) and answers user questions based solely on the content of that document.

RAG Pipeline Overview

This project follows the 5 fundamental steps of a RAG pipeline:

Load: The speech.txt file is loaded into memory using TextLoader.

Split: The document is split into smaller, manageable chunks using RecursiveCharacterTextSplitter.

Embed & Store: Each chunk is converted into a numerical vector (an embedding) using HuggingFaceEmbeddings and stored in a local, in-memory ChromaDB vector store.

Retrieve: When a user asks a question, the system converts the question into an embedding and uses it to find the most semantically similar chunks from the vector store.

Generate: The original question and the retrieved text chunks (as context) are passed to the Ollama LLM (Mistral 7B), which generates a final answer grounded in the provided text.

Technical Stack

Language: Python 3.8+

Framework: LangChain

LLM: Ollama with Mistral 7B (specifically mistral:7b-instruct-q2_K)

Embeddings: HuggingFace sentence-transformers/all-MiniLM-L6-v2

Vector Store: ChromaDB

Option 1: Local Setup & Execution (Recommended)

Follow these steps to run the application on your local machine (e.g., in VS Code).

Step 1: Clone the Repository

Clone this GitHub repository to your local machine:

git clone [https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git](https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git)
cd AmbedkarGPT-Intern-Task


Step 2: Create a Python Environment

It is highly recommended to use a virtual environment (e.g., venv or conda).

Using venv:

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate


Using conda:

conda create -n intern_task python=3.10
conda activate intern_task


Step 3: Create Project Files

This repository should contain main.py and this README.md. You must create the other two required files.

Create requirements.txt:
Create a file named requirements.txt and paste the following dependencies into it:

langchain
langchain-classic
langchain-core
langchain-community
langchain-text-splitters
langchain-ollama
langchain-huggingface
chromadb
sentence-transformers
torch


Create speech.txt:
Create a file named speech.txt and paste the required text into it:

The real remedy is to destroy the belief in the sanctity of the shastras.
How do you expect to succeed if you allow the shastras to continue to be held as sacred and infallible?
You must take a stand against the scriptures. Either you must stop the practice of caste or you must stop believing in the shastras.
You cannot have both. The problem of caste is not a problem of social reform.
It is a problem of overthrowing the authority of the shastras.
So long as people believe in the sanctity of the shastras, they will never be able to get rid of caste.
The work of social reform is like the work of a gardener who is constantly pruning the leaves and branches of a tree without ever attacking the roots.
The real enemy is the belief in the shastras.


Step 4: Install Dependencies

Install all the required Python packages from your requirements.txt file:

pip install -r requirements.txt


Step 5: Set Up Ollama

This application requires the Ollama service to be running locally.

Install Ollama: Download and install the application from https://ollama.ai/

Run Ollama: Ensure the Ollama application is running in the background.

Pull the Model: This code uses the smallest quantized version of Mistral 7B (q2_K) to ensure it runs on most machines. Run the following command in your terminal to pull it:

ollama pull mistral:7b-instruct-q2_K


Step 6: Run the Application

Once your environment is set up and Ollama is running, you can start the Q&A bot:

python main.py


You will see the prompt RAG bot is ready! Type 'quit' to exit..

Example Questions:

What is the real remedy?

What is the work of social reform like?

Option 2: Alternative Setup (Google Colab)

If you have hardware limitations (e.g., low RAM) and cannot run the Mistral 7B model locally, you can use this Google Colab notebook to verify that the code is fully functional.

Simply open a new Google Colab Notebook and paste the entire code block below into a single cell and run it. It will install all dependencies, start Ollama, pull the model, create the files, and run the main.py script automatically.

# --- 1. Install Ollama ---
print("Installing Ollama...")
!curl -fsSL [https://ollama.ai/install.sh](https://ollama.ai/install.sh) | sh

# --- 2. Start Ollama Server & Pull Model ---
print("\nStarting Ollama server and pulling Mistral 7B (q2_K)...")
!nohup ollama serve &
!sleep 5  # Give the server a moment to start
!ollama pull mistral:7b-instruct-q2_K

# --- 3. Install Python Dependencies ---
print("\nInstalling Python dependencies...")
!pip install langchain langchain-classic langchain-core langchain-community langchain-text-splitters chromadb sentence-transformers torch langchain-ollama langchain-huggingface

# --- 4. Create speech.txt File ---
print("\nCreating speech.txt...")
%%writefile speech.txt
The real remedy is to destroy the belief in the sanctity of the shastras.
How do you expect to succeed if you allow the shastras to continue to be held as sacred and infallible?
You must take a stand against the scriptures. Either you must stop the practice of caste or you must stop believing in the shastras.
You cannot have both. The problem of caste is not a problem of social reform.
It is a problem of overthrowing the authority of the shastras.
So long as people believe in the sanctity of the shastras, they will never be able to get rid of caste.
The work of social reform is like the work of a gardener who is constantly pruning the leaves and branches of a tree without ever attacking the roots.
The real enemy is the belief in the shastras.

# --- 5. Create main.py File ---
print("\nCreating main.py...")
%%writefile main.py
"""
Kalpit Pvt Ltd - AI Intern Assignment 1: AmbedkarGPT
This script implements a complete, local-only Retrieval-Augmented Generation (RAG)
pipeline as per the assignment requirements.
"""

import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# === STEP 1: INITIALIZE THE LLM ===
try:
    llm = OllamaLLM(model="mistral:7b-instruct-q2_K")
except Exception as e:
    print(f"Error initializing Ollama. Is the Ollama service running? \nError: {e}")
    sys.exit(1)

# === STEP 2: LOAD AND PROCESS THE DOCUMENT ===
try:
    loader = TextLoader('speech.txt')
    documents = loader.load()
except FileNotFoundError:
    print("Error: speech.txt not found. Please make sure the file is in the same directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading document: {e}")
    sys.exit(1)

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], 
    chunk_size=200,
    chunk_overlap=50,
    length_function=len
)
text_chunks = text_splitter.split_documents(documents)

if not text_chunks:
    print("Error: No text chunks were created. Check speech.txt and splitter settings.")
    sys.exit(1)

# === STEP 3: CREATE EMBEDDINGS AND VECTOR STORE ===
print("Creating vector store with ChromaDB...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

try:
    vector_store = Chroma.from_documents(
        documents=text_chunks, 
        embedding=embedding_model
    )
except Exception as e:
    print(f"Error creating Chroma vector store: {e}")
    sys.exit(1)

retriever = vector_store.as_retriever()
print("Chroma vector store created.")

# === STEP 4: CREATE THE RAG CHAIN (RETRIEVE & GENERATE) ===
qa_system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
The answer must come *only* from the provided text.
Use three sentences maximum and keep the answer concise.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# === STEP 5: RUN THE INTERACTIVE Q&A LOOP ===
print("\nRAG bot is ready! Type 'quit' to exit.")
while True:
    try:
        user_input = input("\nAsk a question about your documents: ").strip()
        if user_input.lower() == 'quit':
            break
        if not user_input:
            continue
        
        response = rag_chain.invoke({"input": user_input})
        print(f"\nAnswer: {response['answer']}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        # Break loop on Colab for a clean run
        break
    except KeyboardInterrupt:
        print("\nExiting...")
        break

# --- 6. Run the main.py script ---
print("\n--- Running the Application ---")
!python main.py
