import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import glob
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# ChromaDB connection
db_path = os.getenv("databasepath")
db_client = chromadb.PersistentClient(path=db_path)
folder_path = os.path.join(os.getenv("20casedocs"), "*.txt")

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

def query_chroma_collection(chroma_collection, query_text, n_results=1):
    """Retrieve relevant legal documents from ChromaDB."""
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name="text-embedding-ada-002"
    )
    
    query_embedding = embedding_function([query_text])[0]
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    if results["documents"]:
        return results["metadatas"][0]  # Return retrieved metadata
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

