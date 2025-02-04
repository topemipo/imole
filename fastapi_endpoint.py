from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from query_db_functions import (
    augment_query_generated, query_chroma_collection, get_file_contents,
    summarize_text_with_map_reduce, generate_response
)

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
db_path = os.getenv("databasepath")
folder_path = os.path.join(os.getenv("20casedocs"), "*.txt")

# Initialize ChromaDB client
db_client = chromadb.PersistentClient(path=db_path)
# Connect to ChromaDB
embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-ada-002"
)

legal_case_collection = db_client.get_collection(
        name="case-docs-collection",
        embedding_function=embedding_function
    )

# FastAPI App
app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

@app.get("/")
def root():
    return {"message": "Legal AI Backend is running!"}

@app.post("/chat/")
def chat(request: QueryRequest):
    """Handles user legal questions, augments query, retrieves case law, and generates AI response."""
    try:
        # Expand the query for better retrieval
        augmented_query = augment_query_generated(request.user_query)
        final_query = f"{request.user_query} {augmented_query}"

        # Query ChromaDB for relevant legal documents
        retrieved_chunks = query_chroma_collection(legal_case_collection, final_query, n_results=1)

        if not retrieved_chunks:
            ai_response = "⚠️ No relevant legal documents found. Consider consulting a lawyer."
            return {"response": ai_response, "retrieved_case": None, "summary": None}

        # Retrieve and summarize the case document
        case_doc = retrieved_chunks[0]["source"]
        file_content = get_file_contents(folder_path, case_doc)
        summary = summarize_text_with_map_reduce(file_content, max_length=1000)
        
        # Generate AI response
        ai_response = generate_response(request.user_query, summary)

        return {"response": ai_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))