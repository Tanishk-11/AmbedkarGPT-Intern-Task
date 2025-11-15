# AmbedkarGPT-Intern-Task

This project is a submission for the Kalpit Pvt Ltd AI Intern Assignment. It is a simple, command-line Q&A system that implements a complete Retrieval-Augmented Generation (RAG) pipeline using LangChain, Ollama, and ChromaDB.

The system ingests a provided text file (`speech.txt`) and answers user questions based *solely* on the content of that document.

## RAG Pipeline Overview

This project follows the 5 fundamental steps of a RAG pipeline:

1.  **Load:** The `speech.txt` file is loaded into memory using `TextLoader`.
2.  **Split:** The document is split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
3.  **Embed & Store:** Each chunk is converted into a numerical vector (an embedding) using `HuggingFaceEmbeddings` and stored in a local, in-memory `ChromaDB` vector store.
4.  **Retrieve:** When a user asks a question, the system converts the question into an embedding and uses it to find the most semantically similar chunks from the vector store.
5.  **Generate:** The original question and the retrieved text chunks (as context) are passed to the `Ollama` LLM (`Mistral 7B`), which generates a final answer grounded in the provided text.

## Technical Stack

* **Language:** Python 3.8+
* **Framework:** LangChain
* **LLM:** Ollama with Mistral 7B (specifically `mistral:7b-instruct-q2_K`)
* **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Store:** ChromaDB

---

## Option 1: Local Setup & Execution (Recommended)

Follow these steps to run the application on your local machine (e.g., in VS Code).

### Step 1: Clone the Repository

Clone this GitHub repository to your local machine (making sure to replace `YOUR_USERNAME` with your GitHub username) and navigate into the new project folder.

### Step 2: Create a Python Environment

It is highly recommended to use a virtual environment. Create a new environment using your preferred tool (like `venv` or `conda`) and activate it.

### Step 3: Create Project Files

This repository should contain `main.py` and this `README.md`. You must create the other two required files.

1.  **Create `requirements.txt`:**
    Create a file named `requirements.txt`. Inside this file, you will need to list all the required Python packages: `langchain`, `langchain-classic`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-ollama`, `langchain-huggingface`, `chromadb`, `sentence-transformers`, and `torch`.

2.  **Create `speech.txt`:**
    Create a file named `speech.txt` and paste the provided speech excerpt into it.

### Step 4: Install Dependencies

Install all the required Python packages by running the `pip install -r requirements.txt` command in your terminal.

### Step 5: Set Up Ollama

This application requires the Ollama service to be running locally.

1.  **Install Ollama:** Download and install the application from the official Ollama website.
2.  **Run Ollama:** Ensure the Ollama application is running in the background.
3.  **Pull the Model:** This code uses the smallest quantized version of Mistral 7B (`q2_K`). You must pull this model by running the `ollama pull mistral:7b-instruct-q2_K` command in your terminal.

### Step 6: Run the Application

Once your environment is set up and Ollama is running, you can start the Q&A bot by executing the main Python script from your terminal: `python main.py`.

You will see the prompt `RAG bot is ready! Type 'quit' to exit.`.

**Example Questions:**
* `What is the real remedy?`
* `What is the work of social reform like?`

---

## Option 2: Alternative Setup (Google Colab)

If you have hardware limitations (e.g., low RAM) and cannot run the Mistral 7B model locally, you can use a Google Colab notebook to verify that the code is fully functional.

The general steps are:
1.  Open a new Google Colab notebook.
2.  In the first cell, install the Ollama service.
3.  In the next cell, start the Ollama server and pull the required Mistral model (`mistral:7b-instruct-q2_K`).
4.  In a new cell, install all the Python dependencies (listed in the `requirements.txt` section above) using `pip install`.
5.  Use Colab's `%%writefile` magic command to create the `speech.txt` file and paste the speech text into it.
6.  Use `%%writefile` again to create the `main.py` file and paste the entire Python script into it.
7.  Finally, run the script by executing `!python main.py` in a new cell to verify the RAG pipeline works.