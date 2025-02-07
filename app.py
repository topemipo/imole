from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import re
import json

from query_db_functions import (
    augment_query_generated, query_postgres_collection, get_file_contents_from_spaces, generate_response,
    preprocess_and_count_tokens,summarize_case_document, summarize_long_document, summarization_pipeline
)


# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_KEY")

# PostgreSQL Connection String
database_url = os.getenv("DATABASE_URL")

# Spaces credentials
spaces_key = os.getenv("SPACES_ACCESS_KEY")
spaces_secret = os.getenv("SPACES_SECRET_KEY")
spaces_bucket = os.getenv("SPACES_BUCKET_NAME")

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

        # Query PostgreSQL (`pgvector`) for relevant legal documents
        retrieved_chunks = query_postgres_collection(final_query, n_results=1)

        if not retrieved_chunks:
            ai_response = "⚠️ No relevant legal documents found. Consider consulting a lawyer."
            return {"response": ai_response, "retrieved_case": None, "summary": None}

        # Retrieve and summarize the case document
        case_doc = retrieved_chunks[0]["source"]
        file_content = get_file_contents_from_spaces(case_doc)
        summary = summarization_pipeline(file_content)

        # Generate AI response
        ai_response = generate_response(request.user_query, summary)

        return {"response": ai_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))