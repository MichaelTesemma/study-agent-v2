# Study Agent

Study from PDFs and quiz yourself using AI -- all running locally on your machine.

## What It Does

- Load PDFs and extract their content
- Store and search knowledge using a local vector database
- Generate quizzes and explanations with Ollama (a local AI model)
- Everything runs on your computer -- no internet or API keys needed

## Prerequisites

1. **Python 3.10 or newer** -- check with `python3 --version`
2. **Ollama** -- download from [ollama.com](https://ollama.com)

## Setup

### Step 1: Install Ollama and Pull the Model

```bash
ollama pull llama3.2
```

### Step 2: Run the Setup Script

```bash
bash setup.sh
```

## How to Run

```bash
python3 server.py
```

Then open **http://localhost:8000** in your browser.

## Troubleshooting

**"Ollama is not running"** -- Make sure Ollama is started. On Mac, open the Ollama app or run `ollama serve` in a terminal.

**"Module not found"** -- Run `bash setup.sh` again to install dependencies.
