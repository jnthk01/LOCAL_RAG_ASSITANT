# LOCAL_RAG_ASSITANT

This project demonstrates a **Retriever-Augmented Generation (RAG)** model using **LangChain** to retrieve relevant information from uploaded PDF files and answer user queries in a conversational format. The model uses the **Gemini API** for language generation and **Google Generative AI embeddings** for vectorizing document content.

## Features
- Upload multiple PDF files for processing.
- The system splits documents into chunks and embeds them for efficient search.
- Users can input queries, and the system will retrieve relevant answers based on the uploaded content.
- Provides answers with context, including the file name and page number from where the information was retrieved.

---

## Installation & Setup

### Step 1: Clone the Repository

Start by cloning this repository to your local machine. In your terminal, run:

```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Set Up a Virtual Environment (Optional but Recommended)

It is recommended to create a virtual environment to isolate your project dependencies.

#### On Windows:
1. Open the command prompt (CMD) or PowerShell.
2. Run the following command to create a virtual environment:

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    ```bash
    venv\Scripts\activate
    ```

#### On macOS/Linux:
1. Open the terminal.
2. Run the following command to create a virtual environment:

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

---

### Step 3: Install Dependencies

Once your virtual environment is activated, you can install all the necessary dependencies using pip. Run the following command:

```bash
pip install -r requirements.txt
```

### Step 4: Set Up the `.env` File

To use the Gemini API and Langsmith API, you'll need to create a `.env` file to store your API keys. Create a file named `.env` in the root directory of the project and add the following lines:

```bash
GEMINI_API_KEY=<your-gemini-api-key>
LANGSMITH_API_KEY=<your-langsmith-api-key>
```

### Step 5: Obtain API Keys

You need to obtain API keys for the following services:
- **Gemini API**: Sign up for an account on Google Cloud or your Gemini provider and generate your API key.
- **Langsmith API**: Sign up for Langsmith and generate your API key.

Once you've obtained the keys, store them in the `.env` file as instructed in Step 4.

---

### Step 6: Generate the `requirements.txt` File

If `requirements.txt` does not exist yet, you can generate it manually or by running the following command inside your virtual environment:

```bash
pip freeze > requirements.txt
