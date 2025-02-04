import os
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import glob
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
import numpy as np

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# database connection
folder_path = os.path.join(os.getenv("CASEDOCS_PATH"), "*.txt")
database_url = os.getenv("DATABASE_URL")

def augment_query_generated(user_query, model="gpt-3.5-turbo"):
    """Generate an augmented query to improve retrieval."""
    system_prompt = """You are a helpful expert legal research assistant. 
    Provide a plausible example answer to the user's query as if you found it in a case document."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

def query_postgres_collection(query_text, n_results=1):
    """Retrieve relevant legal documents from PostgreSQL using pgvector similarity search."""
    
    # Generate embedding for the query
    embedding_response = client.embeddings.create(
        input=[query_text],
        model="text-embedding-ada-002"
    )
    query_embedding = embedding_response.data[0].embedding  # Extract embedding vector
    query_embedding = np.array(query_embedding).tolist()  # Convert to JSON-compatible list

    # Connect to PostgreSQL
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # Perform similarity search using cosine distance with explicit vector casting
    cur.execute("""
        SELECT id, text, source, token_count,
        1 - (embedding <=> %s::vector) AS similarity
        FROM case_docs
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, n_results))

    results = cur.fetchall()
    
    # Close database connection
    cur.close()
    conn.close()

    if results:
        return [{"source": row[2], "text": row[1]} for row in results]  # Return relevant text data
    return []

def get_file_contents(pattern, target_filename):
    """Fetch the full text of a retrieved document."""
    matching_files = glob.glob(pattern)
    for fpath in matching_files:
        if os.path.basename(fpath) == target_filename:
            with open(fpath, 'r', encoding='utf-8') as file:
                return file.read()
    return "Document not found."

def summarize_text_with_map_reduce(file_content: str, max_length: int = 1000) -> str:
    """Summarize a document using LangChain's MapReduce summarization."""
    match = re.search(r"^(.*?)(\n\s*1\.)", file_content, re.DOTALL)
    if match:
        preserved_section = match.group(1).strip()
        remaining_content = file_content[len(preserved_section):].strip()
    else:
        preserved_section = file_content.strip()
        remaining_content = ""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=50  
    )
    chunks = text_splitter.split_text(remaining_content)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    result = chain.invoke({"input_documents": documents, "length": max_length})
    
    summary = preserved_section + "\n\n" + result["output_text"]
    return summary


def generate_response(question, context_data):
    """Generate a final response based on retrieved case documents."""
    prompt = f"""You are a legal assistant designed to help users understand their legal situations by retrieving and summarizing relevant cases. Follow these steps STRICTLY:
    
    1. **Sympathize with the user** (1-2 sentences):
       - Acknowledge their situation with empathy (e.g., "I’m sorry to hear...", "This sounds difficult...").
    
    2. **Retrieve and summarize a case** from the knowledge base below:
    {context_data}
       - Format:
         **Case Name**: [Exact case title]<br>
         **Introduction**: [1-2 sentence overview: who was involved and the core issue]<br>
         **Details**: [Key facts/events in chronological order]<br>
         **Verdict**: [Court decision + outcomes like damages or policy changes]

    3. **Next Steps** (3-4 bullet points):
       - Practical actions tied to the case (e.g., "Save emails from [date range]")
       - Resources (e.g., "Contact [Agency Name] within [timeframe]")
    
    Tone Rules:
    - Professional but compassionate
    - Zero legal jargon (avoid terms like "plaintiff" or "motion")
    - If no matching case: 
      * Apologize briefly
      * Provide 2-3 general steps
      * Add: "Every case is unique – consulting a lawyer is recommended"

    Example structure to mimic:
    "I’m sorry to hear about your situation. Let me share a similar case:
    **Case Name**: Smith v. ABC Corp
    **Introduction**: A warehouse worker fired after reporting safety issues.
    **Details**: The employee reported violations in March 2022, was terminated April 2022 with no warning. The employer claimed budget cuts.
    **Verdict**: Court ruled wrongful termination – $150k awarded due to retaliation evidence.
    Next steps:
    - Document all safety reports you filed
    - Contact OSHA within 30 days
    - Consult an employment lawyer"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=1500
    )

    return response.choices[0].message.content

