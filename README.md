# 🔍 docsec (DocScan) - PDF Security Analyzer

Verifying documents before sharing for security risks.

This is a complete, production-ready Streamlit application that performs advanced PDF text extraction, intelligent multi-word keyword scanning, TF-IDF topic analysis, and local AI security analysis using Ollama. All processing is 100% local, ensuring no sensitive data leaves your machine.

## Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama Setup (for AI Insights)**:
   - Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh` (or download from ollama.ai for Windows)
   - Serve Ollama in a terminal: `ollama serve`
   - Open a new terminal and pull the required model: `ollama pull llama3.2:3b`

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Features

- **Keyword Scanner**: Accurately scans PDFs for highly sensitive keywords using flexible multi-word phrase matching, capturing the immediate context around each match.
- **Cross-Document Analysis (TF-IDF)**: Automatically determines what each document is about by comparing the relative frequencies of terms across your uploaded documents.
- **AI Security Insights**: Integrates directly with your local Ollama instance to analyze document excerpts, detect primary risks, find sensitive data types, and provide actionable security recommendations before sharing.
