# How to Run AmbedkarGPT

This file provides two methods to run this project: locally on your own machine or on Google Colab.

---

## Option 1: Local Setup & Execution

Follow these steps to run the application on your local computer.

### Step 1: Set Up the Project
First, clone this repository to your computer and navigate into the project folder.

### Step 2: Create a Python Environment
It is highly recommended to use a virtual environment. You can create one using `venv` or `conda` and then activate it.

### Step 3: Install Dependencies
Once your environment is active, you must install all the required Python packages. All packages are listed in the `requirements.txt` file. You can install them by running `pip install -r requirements.txt` in your terminal.

### Step 4: Set Up Ollama
This application requires the Ollama service to run the language model locally.

1.  **Install Ollama:** Go to the official Ollama website (`ollama.ai`) and download and install the application for your operating system.
2.  **Run Ollama:** After installation, make sure the Ollama application is running in the background.
3.  **Pull the Model:** You must download the specific Mistral 7B model required for this project. Open your terminal and run the command `ollama pull mistral:7b-instruct-q2_K`.

### Step 5: Run the Application
With your environment activated, all dependencies installed, and Ollama running, you can now start the Q&A bot.

From your terminal, run the command `python main.py`.

The script will start, and you will see a message "RAG bot is ready! Type 'quit' to exit." You can now ask questions based on the provided text.

---

## Option 2: Alternative Setup (Google Colab)

If your local computer does not have enough RAM to run the Mistral 7B model, you can use Google Colab as a free alternative to verify that the code is fully functional.

### Step 1: Open a New Notebook
Go to Google Colab and create a new, blank notebook.

### Step 2: Install and Run Ollama
In the first code cell, you need to install and start the Ollama service. You can do this by running the curl command to install it, then running `ollama serve` in the background, and finally using `ollama pull mistral:7b-instruct-q2_K` to download the required model.

### Step 3: Install Python Dependencies
In the next code cell, you must install all the Python packages listed in the `requirements.txt` file using `pip install`.

### Step 4: Create Project Files
You will need to create the `speech.txt` and `main.py` files inside the Colab environment. You can do this by using the `%%writefile` magic command.

1.  Create a new cell and use `%%writefile speech.txt`. On the following lines, paste the full text of the speech excerpt.
2.  Create another new cell and use `%%writefile main.py`. On the following lines, paste the entire contents of your `main.py` script.

### Step 5: Run the Project
Finally, in a new code cell, run the command `!python main.py`. This will execute your script within the Colab environment, which has enough RAM to load the model. You will see the script initialize and answer a test question, proving the pipeline works.