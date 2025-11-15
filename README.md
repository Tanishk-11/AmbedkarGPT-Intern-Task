AmbedkarGPT — Intern Task (Kalpit Pvt Ltd)

A fully functional Retrieval-Augmented Generation (RAG) pipeline built using LangChain, Ollama, and ChromaDB, designed as part of the Kalpit Pvt Ltd AI Internship Assignment.

The system loads a document (speech.txt), embeds it, stores vectors in ChromaDB, retrieves relevant chunks based on the query, and generates accurate answers using Mistral 7B (via Ollama).

🚀 RAG Pipeline Overview

The project uses the 5 core steps of a RAG pipeline:

1️⃣ Load

Load the text document (speech.txt) using TextLoader.

2️⃣ Split

Break the document into small chunks using RecursiveCharacterTextSplitter.

3️⃣ Embed & Store

Convert chunks into embeddings using HuggingFace (all-MiniLM-L6-v2) and store them in ChromaDB.

4️⃣ Retrieve

Retrieve the top semantically similar chunks when the user asks a question.

5️⃣ Generate

Feed the question + context into Ollama LLM (Mistral 7B) to generate an answer grounded only in the provided document.

🛠️ Tech Stack
Component	Tool
Language	Python 3.8+
RAG Framework	LangChain
LLM	Ollama (Mistral 7B — mistral:7b-instruct-q2_K)
Embeddings	HuggingFace Sentence Transformers
Vector Store	ChromaDB
Text Splitting	Langchain Text Splitters
🚀 Option 1: Local Setup (Recommended)
Step 1 — Clone the Repository
git clone https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task

Step 2 — Create Virtual Environment
Using venv
python -m venv venv
source venv/bin/activate       # Windows: .\venv\Scripts\activate

Using Conda
conda create -n intern_task python=3.10
conda activate intern_task

Step 3 — Required Files
Create requirements.txt:
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
The real remedy is to destroy the belief in the sanctity of the shastras.
How do you expect to succeed if you allow the shastras to continue to be held as sacred and infallible?
You must take a stand against the scriptures. Either you must stop the practice of caste or you must stop believing in the shastras.
You cannot have both. The problem of caste is not a problem of social reform.
It is a problem of overthrowing the authority of the shastras.
So long as people believe in the sanctity of the shastras, they will never be able to get rid of caste.
The work of social reform is like the work of a gardener who is constantly pruning the leaves and branches of a tree without ever attacking the roots.
The real enemy is the belief in the shastras.

Step 4 — Install Dependencies
pip install -r requirements.txt

Step 5 — Set Up Ollama
1. Install Ollama:

Download from → https://ollama.ai/

2. Ensure Ollama is running in background
3. Pull the Mistral 7B (q2_K) model:
ollama pull mistral:7b-instruct-q2_K

Step 6 — Run the Application
python main.py


You will see:

RAG bot is ready! Type 'quit' to exit.

Example queries:

What is the real remedy?

What is the work of social reform like?

☁️ Option 2: Run on Google Colab

If your system cannot run Mistral locally, use this one-cell Colab script.

Paste this entire block into a single Colab cell and run it:

# --- 1. Install Ollama ---
print("Installing Ollama...")
!curl -fsSL https://ollama.ai/install.sh | sh

# --- 2. Start Ollama Server & Pull Model ---
print("\nStarting Ollama server and pulling Mistral 7B (q2_K)...")
!nohup ollama serve &
!sleep 5
!ollama pull mistral:7b-instruct-q2_K

# --- 3. Install Python Dependencies ---
print("\nInstalling Python dependencies...")
!pip install langchain langchain-classic langchain-core langchain-community langchain-text-splitters chromadb sentence-transformers torch langchain-ollama langchain-huggingface

# --- 4. Create speech.txt ---
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

# --- 5. Create main.py ---
print("\nCreating main.py...")
%%writefile main.py
<insert your main.py content exactly as written in your project>

# --- 6. Run the app ---
print("\n--- Running the Application ---")
!python main.py


Replace <insert your main.py content...> with your actual script.

📁 Folder Structure
AmbedkarGPT-Intern-Task/
│── main.py
│── speech.txt
│── requirements.txt
│── README.md   <-- (this file)

📌 Final Notes

The model only answers using content from speech.txt.

If the answer is not present in the text, it will respond:
“I don’t know.”

This ensures proper grounded RAG behavior.

🙌 Author

This project was created as part of the Kalpit Pvt Ltd AI Internship Assignment.

If you need help with improving your RAG, deploying with Docker, making it a FastAPI app, or adding UI, feel free to ask!