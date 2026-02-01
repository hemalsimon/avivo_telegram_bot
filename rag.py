import os
import glob
import sqlite3
import numpy as np
import ollama
import time
import pickle
from functools import lru_cache
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self, docs_dir="data/docs", db_path="rag.db", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the RAG system with SQLite persistence.
        """
        self.docs_dir = docs_dir
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(model_name)
        
        self.embeddings = None
        self.chunks = []
        
        # Load local Ollama model from env or default
        self.llm_model = os.getenv("OLLAMA_MODEL", "kimi-k2.5:cloud")
        print(f"RAG Initialized with LLM: {self.llm_model}")

        # Initialize DB and Index
        self._init_db()
        self._build_index()

    def _init_db(self):
        """Initialize the SQLite database for chunks and embeddings."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                chunk_text TEXT,
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()

    def _load_documents(self):
        """Read all .md and .txt files from the docs directory."""
        files = glob.glob(os.path.join(self.docs_dir, "**/*.md"), recursive=True) + \
                glob.glob(os.path.join(self.docs_dir, "**/*.txt"), recursive=True)
        documents = []
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                documents.append({"path": fpath, "content": f.read()})
        return documents

    def _chunk_text(self, text, chunk_size=500, overlap=50):
        """Simple overlapping chunker."""
        text_chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            text_chunks.append(chunk)
            start += (chunk_size - overlap)
        return text_chunks

    def _build_index(self):
        """Load from DB if available, else process docs and save to DB."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Check if we have data
        cur.execute("SELECT count(*) FROM chunks")
        count = cur.fetchone()[0]

        if count > 0:
            print(f"Loading {count} chunks from SQLite database '{self.db_path}'...")
            cur.execute("SELECT source, chunk_text, embedding FROM chunks")
            rows = cur.fetchall()
            
            self.chunks = []
            embeddings_list = []
            
            for row in rows:
                source, text, blob = row
                self.chunks.append({"source": source, "text": text})
                embeddings_list.append(pickle.loads(blob))
            
            self.embeddings = np.array(embeddings_list)
            print("Knowledge Base loaded from DB.")
        
        else:
            print("No DB data found. Processing documents...")
            raw_docs = self._load_documents()
            self.chunks = []
            
            # Helper list for DB insertion
            db_data = []

            for doc in raw_docs:
                doc_chunks = self._chunk_text(doc["content"])
                for chunk in doc_chunks:
                    source_name = os.path.basename(doc["path"])
                    self.chunks.append({
                        "text": chunk,
                        "source": source_name
                    })
            
            if not self.chunks:
                print("Warning: No documents found to index.")
                self.embeddings = np.array([])
            else:
                print(f"Computing embeddings for {len(self.chunks)} chunks...")
                texts = [c["text"] for c in self.chunks]
                self.embeddings = self.embedding_model.encode(texts)
                
                # Insert into DB
                print("Saving to SQLite...")
                for i, chunk in enumerate(self.chunks):
                    emb_blob = pickle.dumps(self.embeddings[i])
                    db_data.append((chunk["source"], chunk["text"], emb_blob))
                
                cur.executemany("INSERT INTO chunks (source, chunk_text, embedding) VALUES (?, ?, ?)", db_data)
                conn.commit()
                print("Knowledge Base saved to DB.")

        conn.close()

    @lru_cache(maxsize=100)
    def retrieve(self, query, k=3):
        """Retrieve top-k relevant chunks using in-memory cosine similarity. Cached."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_embedding = self.embedding_model.encode([query])[0]
        scores = np.dot(self.embeddings, query_embedding)
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append(self.chunks[idx])
        return results

    def generate_answer(self, query, history=""):
        """Full RAG pipeline with Auto-Retry and History."""
        # 1. Retrieve
        relevant_chunks = self.retrieve(query)
        sources = []
        if not relevant_chunks:
             context_text = "No relevant context found."
        else:
            context_text = "\n\n---\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in relevant_chunks])
            sources = list(set([c['source'] for c in relevant_chunks]))

        prompt = f"""You are a helpful assistant for a financial institution. 
Answer the user's question using ONLY the provided context below.
If the answer is not in the context, say "I don't have information on that."

Conversation History:
{history}

Context:
{context_text}

Question: {query}
Answer:"""

        # 3. Generation via Ollama
        for attempt in range(3):
            try:
                response = ollama.chat(model=self.llm_model, messages=[
                    {'role': 'user', 'content': prompt},
                ])
                return response['message']['content'], sources
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    print(f"Ollama busy (503), retrying in 2s... (Attempt {attempt+1}/3)")
                    time.sleep(2)
                    continue
                return f"Error contacting LLM: {str(e)}", []

if __name__ == "__main__":
    rag = RAGSystem()
    print("Testing Retrieval...")
    print(rag.generate_answer("What is the asset allocation for stocks?"))
