import os
import glob
import psycopg2
import numpy as np
from psycopg2.extras import execute_values
from openai import OpenAI
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# PostgreSQL Connection String
database_url = os.getenv("DATABASE_URL")

# Initialize OpenAI client
client = OpenAI(api_key=openai_key)

def load_text_files(folder_path):
    """Loads all text files from a given directory."""
    all_texts = []
    for filepath in glob.glob(folder_path):
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    all_texts.append({"text": text, "source": os.path.basename(filepath)})
    return all_texts

def chunk_texts(all_texts, chunk_size=8100, chunk_overlap=500):
    """Splits a list of text entries into chunks using a token-aware text splitter."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len(tokenizer.encode(text)),
        is_separator_regex=False
    )
    chunked_texts = []
    for doc_index, entry in enumerate(all_texts):
        chunks = splitter.split_text(entry["text"])
        for chunk_index, chunk in enumerate(chunks):
            chunked_texts.append({
                "id": f"doc_{doc_index}_chunk_{chunk_index}",
                "text": chunk,
                "metadata": {
                    "source": entry["source"],
                    "token_count": len(tokenizer.encode(chunk))
                }
            })
    return chunked_texts

def add_to_postgres(chunked_texts):
    """Stores chunked texts and embeddings in PostgreSQL (pgvector)."""
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # Ensure the table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_docs (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            source TEXT,
            token_count INT,
            embedding VECTOR(1536)  -- OpenAI embeddings size
        );
    """)

    # Generate embeddings
    data_to_insert = []
    for item in chunked_texts:
        embedding_response = client.embeddings.create(
            input=[item["text"]],
            model="text-embedding-ada-002"
        )
        vector = embedding_response.data[0].embedding  # OpenAI returns a list of embeddings
        vector_str = np.array(vector).tolist()  # Convert to JSON-compatible format
        
        data_to_insert.append((
            item["id"],
            item["text"],
            item["metadata"]["source"],
            item["metadata"]["token_count"],
            vector_str
        ))

    # Insert into PostgreSQL
    execute_values(cur, """
        INSERT INTO case_docs (id, text, source, token_count, embedding)
        VALUES %s ON CONFLICT (id) DO NOTHING;
    """, data_to_insert)

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Successfully stored {len(data_to_insert)} chunks in PostgreSQL.")

def process_documents():
    """Main function to load, process, and store documents."""
    folder_path = os.path.join(os.getenv("CASEDOCS_PATH"), "*.txt")
    all_texts = load_text_files(folder_path)
    chunked_texts = chunk_texts(all_texts)
    add_to_postgres(chunked_texts)

if __name__ == "__main__":
    process_documents()