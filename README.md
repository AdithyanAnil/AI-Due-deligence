# AI Due Diligence Copilot

An AI-powered document analysis application that helps users review business documents, extract important information, investigate potential risks, and generate structured insights using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

## Overview

Traditional due diligence involves reviewing lengthy documents such as annual reports, financial statements, investor presentations, and market research reports. This process can be time-consuming and makes it difficult to locate specific information across multiple documents.

The **AI Due Diligence Copilot** simplifies this process by allowing users to upload PDF documents, ask questions about their contents, analyze relevant findings, and generate PDF reports through an interactive web application.

## Key Features

- **Multi-PDF Upload:** Upload multiple business documents for analysis.
- **Automated Text Extraction:** Extract text from uploaded PDF documents.
- **Document Chunking:** Split extracted text into smaller chunks for efficient retrieval.
- **Semantic Search:** Retrieve relevant document content using vector embeddings.
- **AI-Powered Question Answering:** Ask natural-language questions about uploaded documents.
- **Financial Analysis:** Examine relevant financial information and performance indicators.
- **Risk Identification:** Investigate potential business and financial risks.
- **Growth Opportunity Analysis:** Explore opportunities identified in the documents.
- **Contradiction Detection:** Assist in identifying inconsistencies across documents.
- **Follow-up Investigation:** Generate questions for further due diligence.
- **PDF Report Generation:** Export analysis findings into a downloadable report.
- **Interactive Web Interface:** Access the application through Streamlit.

*Feature availability depends on the implemented modules and the information contained in the uploaded documents.*

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Streamlit | Interactive web interface |
| LangChain | Document processing and RAG orchestration |
| Hugging Face Embeddings | Convert text chunks into vector representations |
| ChromaDB | Store and retrieve document embeddings |
| OpenAI API | Generate AI responses using retrieved context |
| PyPDF / PyMuPDF / PyPDFLoader | PDF text extraction, depending on implementation |
| ReportLab | PDF report generation, if used in the project |

## System Architecture

```text
           User
            |
            v
     Streamlit Interface
            |
            v
      Upload PDF Files
            |
            v
     Extract Document Text
            |
            v
      Split into Chunks
            |
            v
  Hugging Face Embeddings
            |
            v
     Chroma Vector Store
            |
            v
       User Question
            |
            v
     Semantic Retrieval
            |
            v
       Relevant Context
            |
            v
        OpenAI API
            |
            v
     Generated Response
            |
            v
     User / PDF Report
```

## How It Works

### 1. Document Upload
Users upload one or more PDF documents through the Streamlit interface.

### 2. Text Extraction
The application extracts readable text from the uploaded documents.

### 3. Text Chunking
The extracted text is divided into smaller chunks to improve retrieval accuracy and fit the model's context requirements.

### 4. Embedding Generation
Hugging Face embedding models convert document chunks into numerical vector representations.

### 5. Vector Storage
The generated embeddings and associated document content are stored in ChromaDB.

### 6. Retrieval-Augmented Generation
When a user submits a question, the retriever searches for semantically relevant document chunks. LangChain passes the retrieved context and question to the OpenAI language model.

### 7. Response Generation
The OpenAI API generates a response using the retrieved context to help answer the user's question.

### 8. Report Generation
Users can generate a PDF report of their analysis using the report-generation module.

## Project Structure

The following is an example structure. Adjust the filenames to match your actual project.

```text
AI Due Diligence/
│
├── app.py
├── report.py
├── requirements.txt
├── .env
├── .gitignore
│
├── src/
│   ├── loader.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── rag_chain.py
│
├── data/
│   └── uploads/
│
└── chroma_db/
```

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version supported by your dependencies
- An OpenAI API key
- Internet access for API calls and model downloads
- Visual Studio Code or another Python development environment

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd "AI Due Diligence"
```

Replace `<your-repository-url>` with your actual repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If you have not created `requirements.txt`, generate it from your working environment:

```bash
pip freeze > requirements.txt
```

Review the generated file before committing it to your repository.

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Replace the placeholder with your own API key. Never commit your actual API key to GitHub.

Ensure that your application loads the environment variables, for example, using `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## Example Use Cases

- Reviewing annual reports and financial statements.
- Extracting relevant information from investor presentations.
- Investigating potential financial and operational risks.
- Comparing information across multiple business documents.
- Preparing preliminary due diligence summaries.
- Generating reports for further review.

## Security and Limitations

- Keep API keys in environment variables and exclude `.env` from version control.
- Treat uploaded documents as potentially confidential and handle them appropriately.
- Validate AI-generated responses against the original source documents.
- Ensure source references are available before treating a response as a verified finding.
- AI-generated analysis may contain errors or omit important information.
- Financial and legal decisions require independent verification and professional judgment.
- Remove temporary uploads and vector-store data when no longer needed, according to the application's cleanup configuration.

## Future Enhancements

- Multi-agent architecture for specialized financial, legal, and risk analysis.
- More reliable source citations with page-level references.
- Advanced financial ratio calculations and visualization.
- Improved cross-document contradiction detection.
- OCR support for scanned PDFs.
- Custom report templates and enhanced PDF formatting.
- Secure deployment with user authentication and access controls.

## Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG).
- Large Language Model integration through the OpenAI API.
- Semantic search and vector databases.
- Document processing and text embeddings.
- LLM-powered business document analysis.
- Interactive AI application development with Streamlit.

## Disclaimer

The AI Due Diligence Copilot is intended to assist with preliminary document analysis. It does not replace professional financial, legal, accounting, or investment advice. All findings should be independently verified before being used for consequential decisions.

## Author

**Adithyan S Anil**

Data Science and AI/ML

Developed using Python, Streamlit, LangChain, ChromaDB, Hugging Face Embeddings, and the OpenAI API.

