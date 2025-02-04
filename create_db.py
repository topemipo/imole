import os
import glob
from dotenv import load_dotenv

from openai import OpenAI
import tiktoken

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.utils import embedding_functions

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

def load_text_files(folder_path):
    """Loads all text files from a given directory."""
    all_text = []
    for filepath in glob.glob(folder_path):
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    all_text.append({"text": text, "source": os.path.basename(filepath)})
    return all_text

def chunk_casedocs(all_texts, chunk_size=8100, chunk_overlap=500):
    """
    Splits a list of text entries into chunks using a token-aware text splitter.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    optimized_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len(tokenizer.encode(text)),
        is_separator_regex=False
    )
    chunked_texts = []
    for doc_index, entry in enumerate(all_texts):
        chunks = optimized_splitter.split_text(entry["text"])
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

def add_to_chroma_collection(chunked_texts, openai_key, collection_name="case-docs-collection"):
    """Adds chunked texts to a Chroma collection with OpenAI embeddings."""
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name="text-embedding-ada-002"
    )
    chroma_client = chromadb.PersistentClient(path="database")
    existing_collections = [col.name for col in chroma_client.list_collections()]
    if collection_name in existing_collections:
        chroma_collection = chroma_client.get_collection(collection_name)
    else:
        chroma_collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    ids = [item["id"] for item in chunked_texts]
    documents = [item["text"] for item in chunked_texts]
    metadatas = [item["metadata"] for item in chunked_texts]
    chroma_collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    return chroma_collection

def generate_data_store():
    """Main function to load, process, and store data."""
    folder_path = os.path.join(os.getenv("CASEDOCS_PATH"), "*.txt")
    all_texts = load_text_files(folder_path)
    chunked_texts = chunk_casedocs(all_texts)
    add_to_chroma_collection(chunked_texts, openai_key)

def main():
    generate_data_store()

if __name__ == "__main__":
    main()
