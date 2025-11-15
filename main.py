"""
Kalpit Pvt Ltd - AI Intern Assignment 1: AmbedkarGPT
Author: Tanishk
Date: 2025-11-15

This script implements a complete, local-only Retrieval-Augmented Generation (RAG)
pipeline as per the assignment requirements.

The RAG Pipeline Steps:
1.  Load: Load the 'speech.txt' document.
2.  Split: Split the document into manageable chunks.
3.  Embed & Store: Convert chunks to vectors (embeddings) and store them in
    a local ChromaDB vector store.
4.  Retrieve: Based on a user's question, retrieve the most relevant chunks
    from the vector store.
5.  Generate: Pass the user's question and the retrieved chunks (as context)
    to a local LLM (Mistral 7B) to generate a "grounded" answer.
"""

# --- Core System & File Handling ---
import sys

# --- LangChain Imports: Data Loading and Splitting ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# --- LangChain Imports: Embeddings and Vector Store ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- LangChain Imports: LLM, Chains, and Prompts ---
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# === STEP 1: INITIALIZE THE LLM ===
# Load the local LLM (Ollama with Mistral 7B).
# We use the 'mistral:7b-instruct-q2_K' tag, which is the smallest
# quantized version of Mistral 7B. This ensures it can run on
# machines with limited RAM, while still fulfilling the "Mistral 7B" requirement.
try:
    llm = OllamaLLM(model="mistral:7b-instruct-q2_K")
except Exception as e:
    # Fail fast if the Ollama service isn't running in the background.
    print(f"Error initializing Ollama. Is the Ollama service running? \nError: {e}")
    sys.exit(1)

# === STEP 2: LOAD AND PROCESS THE DOCUMENT ===
try:
    # Load the specified text file.
    loader = TextLoader('speech.txt')
    documents = loader.load()
except FileNotFoundError:
    print("Error: speech.txt not found. Please make sure the file is in the same directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading document: {e}")
    sys.exit(1)

# Split the loaded document into smaller chunks.
# RecursiveCharacterTextSplitter is robust, trying to split by
# paragraphs, then new lines, then spaces.
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], 
    chunk_size=200,      # Small chunk size for a very short document
    chunk_overlap=50,    # Overlap helps maintain context between chunks
    length_function=len
)
text_chunks = text_splitter.split_documents(documents)

if not text_chunks:
    print("Error: No text chunks were created. Check speech.txt and splitter settings.")
    sys.exit(1)

# === STEP 3: CREATE EMBEDDINGS AND VECTOR STORE ===
print("Creating vector store with ChromaDB...")

# Initialize the embedding model.
# This uses the specific, free, and local-only model from HuggingFace
# as required by the assignment.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create the Chroma vector store.
# This step does two things:
# 1. It calculates the embeddings (vectors) for all text_chunks.
# 2. It stores these vectors in an in-memory ChromaDB database.
try:
    vector_store = Chroma.from_documents(
        documents=text_chunks, 
        embedding=embedding_model
    )
except Exception as e:
    print(f"Error creating Chroma vector store: {e}")
    sys.exit(1)

# === STEP 4: CREATE THE RAG CHAIN (RETRIEVE & GENERATE) ===

# Create a retriever from the vector store.
# The retriever's job is to fetch relevant documents based on a query.
retriever = vector_store.as_retriever()

# This is the prompt template that "stuffs" the retrieved context
# and the user's question into a single prompt for the LLM.
# This is the core of RAG.
qa_system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
The answer must come *only* from the provided text.
Use three sentences maximum and keep the answer concise.

{context}"""

# Create the LangChain prompt template
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    ("human", "{input}"),
])

# This chain takes the user's question and the retrieved documents
# and combines them into the prompt.
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# This is the final, complete RAG chain.
# It chains the two steps together:
# 1. 'retriever': Takes the user input and retrieves documents.
# 2. 'question_answer_chain': Takes the user input and retrieved documents
#    to generate the final answer.
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# === STEP 5: RUN THE INTERACTIVE Q&A LOOP ===
print("\nRAG bot is ready! Type 'quit' to exit.")
while True:
    try:
        # Get input from the user.
        user_input = input("\nAsk a question about your documents: ").strip()
        
        # Allow the user to exit.
        if user_input.lower() == 'quit':
            break
        if not user_input:
            continue

        # Invoke the RAG chain with the user's question.
        # The chain handles the entire Retrieve -> Generate pipeline.
        response = rag_chain.invoke({"input": user_input})
        
        # Print the answer.
        print(f"\nAnswer: {response['answer']}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
        break