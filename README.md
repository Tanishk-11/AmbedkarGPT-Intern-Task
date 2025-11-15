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

Clone this GitHub repository to your local machine:
```bash
git clone [https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git](https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git)
cd AmbedkarGPT-Intern-Task