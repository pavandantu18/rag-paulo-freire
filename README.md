# RAG Paulo Freire

This project is a Retrieval-Augmented Generation (RAG) system built to answer questions about the works of Paulo Freire and the principles of conscious critical computing.

## Project Goal

The goal of this research is to build and evaluate a RAG agent capable of answering questions based on a specialized knowledge base of texts from Paulo Freire and the field of conscious critical computing. The project tests and compares the performance of three different open-source language models in this task.

### Models Tested

  * `llama3:8b`
  * `deepseek-r1:14b`
  * `gemma3:12b`

-----

## Prerequisites

  * Python 3.11+
  * Git
  * Ollama
  * NVIDIA GPU with CUDA support (e.g., RTX A4500 or better)
  * RAM: A minimum of 48 GB

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/bryanpinheiro/rag-paulo-freire.git
    cd rag-paulo-freire
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Ollama:**
    If you do not have Ollama installed, run the official script.

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

-----

## Running the Application

You will need two terminals to run the application.

1.  **Start the Ollama Server:**
    In your first terminal, start the Ollama service to serve the models.

    ```bash
    ollama serve
    ```

    Leave this terminal running.

2.  **Download the Language Models:**
    In a second terminal, download the language models for testing.

    ```bash
    ollama pull llama3:8b
    ```

    ```bash
    ollama pull deepseek-r1:14b
    ```

    ```bash
    ollama pull gemma3:12b
    ```

3.  **Run the Application:**
    In the second terminal, run the main script.

    ```bash
    python main.py
    ```

    Once running, open the public URL provided in the terminal to access the application.