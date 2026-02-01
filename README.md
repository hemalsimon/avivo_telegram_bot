# Mini-RAG Telegram Bot 🤖💰

A lightweight GenAI bot that answers finance-related questions using a local Retrieval-Augmented Generation (RAG) system. Powered by **Ollama** (`kimi-k2.5:cloud`) and **Python Telegram Bot**.

## 🚀 Features
- **RAG Engine**: Retrieves answers from local finance documents.
- **Persistent Storage**: SQLite (instantly loads chunks on restart).
- **Smart Features**:
    - **Memory**: Context-aware (remembers last 3 messages).
    - **Sources**: Cites specific documents used.
    - **Caching**: Instant responses for repeated queries.
    - **Summarization**: `/summarize` command for conversation summaries.
- **Robustness**: Auto-retry logic for network stability.
- **Privacy-First**: Vector search runs entirely locally (NumPy equivalent).

## 📦 Tech Stack
- **Bot**: `python-telegram-bot`
- **LLM**: Ollama (`kimi-k2.5:cloud`)
- **Embeddings**: `sentence-transformers`
- **Vector Store**: SQLite (Persistence) + NumPy (Search)

## 🛠️ Setup & Run

### 1. Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running.
- Pull the model:
  ```bash
  ollama run kimi-k2.5:cloud
  ```

### 2. Installation
```bash
# Clone or download the repo
cd "s:/rag for avio"

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file (copy from `.env.example`) and add your Telegram Token:
```ini
TELEGRAM_TOKEN=your_token_here
OLLAMA_MODEL=kimi-k2.5:cloud
OLLAMA_BASE_URL=http://localhost:11434
```

### 4. Run the Bot
```bash
python main.py
```
Search for your bot in Telegram and click `/start`.

## 📂 Project Structure
```text
├── main.py                 # Bot entry point
├── rag.py                  # RAG engine (Indxing & Retrieval)
├── data/
│   └── docs/               # Knowledge base (Add your .md/.txt files here)
└── requirements.txt        # Dependencies
```

## 🧪 Testing RAG
To test the RAG engine without the bot:
```bash
python test_rag.py
```
