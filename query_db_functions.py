import os
import re
import json

from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import psycopg2
import numpy as np
import tiktoken
import boto3

# Load environment variables
load_dotenv()

# OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Anthropic
anthropic_key = os.getenv("ANTHROPIC_KEY")
client_atp = anthropic.Anthropic(api_key=anthropic_key)

# database connection
database_url = os.getenv("DATABASE_URL")

# Spaces credentials
spaces_key = os.getenv("SPACES_ACCESS_KEY")
spaces_secret = os.getenv("SPACES_SECRET_KEY")
spaces_bucket = os.getenv("SPACES_BUCKET_NAME")

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

def get_file_contents_from_spaces(file_name, folder_name="casedocs"):
    """Retrieve file content from DigitalOcean Spaces inside a folder."""
    # Initialize Spaces client
    s3_client = boto3.client(
        "s3",
        region_name="lon1",
        endpoint_url=f"https://lon1.digitaloceanspaces.com",
        aws_access_key_id=spaces_key,
        aws_secret_access_key=spaces_secret
    )
    s3_file_path = f"{folder_name}/{file_name}"  # Adjusted for folder structure
    try:
        obj = s3_client.get_object(Bucket=spaces_bucket, Key=s3_file_path)
        content = obj['Body'].read().decode('utf-8')
        return content
    except s3_client.exceptions.NoSuchKey:
        return "Document not found."
    except Exception as e:
        return f"Error retrieving document: {str(e)}"

def preprocess_and_count_tokens(text, model="claude-3.5-sonnet-20241022"):
    """Preprocesses text and counts tokens using an approximate tokenizer."""
    processed_text = " ".join(text.split()).lower()

    # Using an OpenAI tokenizer (approximation)
    encoding = tiktoken.get_encoding("cl100k_base")  # Best approximation for Claude
    token_count = len(encoding.encode(processed_text))

    return token_count

def summarize_case_document(text, model="claude-3-5-sonnet-20241022", max_tokens=1000):

    #remove whitespace from text
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Format the details to extract to be placed within the prompt's context
    details_to_extract = [
    'Case name (official title of the case)',
    'Presiding judge (name of the judge who delivered the verdict)',
    'Case summary (brief description of what the case was about)',
    'Key details (specific facts, arguments, and legal points raised)',
    'Rationale (reasoning and legal principles applied in reaching the verdict)',
    'Verdict (final judgment and outcome of the case)']
    
    details_to_extract_str = '\n'.join(details_to_extract)

    # Prompt the model to summarize the case document
    prompt = f"""Summarize the following legal case document. Focus on these key aspects:

    {details_to_extract_str}

    Provide the summary in bullet points nested within the XML header for each section. For example:
    <case_summary>
        <case_name>
            - [Official Case Name]
        </case_name>

        <presiding_judge>
            - Name: [Judge’s Full Name]
        </presiding_judge>

        <case_summary>
            - Summary: [Brief description of the case]
        </case_summary>

        <key_details>
            - Facts: [Essential facts of the case]
            - Arguments: [Key arguments presented by both sides]
            - Legal Points: [Relevant laws, precedents, or statutes cited]
        </key_details>

        <rationale>
            - Reasoning: [Legal reasoning behind the verdict]
            - Principles Applied: [Judicial principles or frameworks considered]
        </rationale>

        <verdict>
            - Final Judgment: [Outcome of the case]
            - Justification: [Why this verdict was reached]
        </verdict>
    </case_summary>

    If any information is not explicitly stated in the document, note it as "Not specified". Do not include unnecessary preambles.

    Case document text:
    {text}
    """
    response = client_atp.messages.create(
        model=model,
        max_tokens=max_tokens,
        system="You are a legal analyst specialising in case law, known for producing highly accurate and structured summaries of legal precedents.",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Here is the summary of the legal case document: <summary>"}
        ],
        stop_sequences=["</summary>"]
    )

    return response.content[0].text

def summarize_long_document(text, model="claude-3-5-sonnet-20241022", max_tokens=1000, chunk_size=20000):
    """
    Splits a long document into chunks, summarizes each chunk, and then combines the summaries into a final structured summary.
    
    Parameters:
        text (str): The document to be summarized.
        details_to_extract (list): Key aspects to focus on in the summary.
        model (str): AI model used for summarization.
        chunk_size (int): Maximum size of each text chunk.
        max_tokens (int): Maximum tokens for each summary request.
    
    Returns:
        str: A structured summary of the entire document.
    """
    # Remove whitespace and numbering artifacts
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Chunk the text
    chunk_text = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Format the details to extract
    details_to_extract = [
    'Case name (official title of the case)',
    'Presiding judge (name of the judge who delivered the verdict)',
    'Case summary (brief description of what the case was about)',
    'Key details (specific facts, arguments, and legal points raised)',
    'Rationale (reasoning and legal principles applied in reaching the verdict)',
    'Verdict (final judgment and outcome of the case)']
    details_to_extract_str = '\n'.join(details_to_extract)
    
   # Iterate over chunks and summarize each one
    chunk_summaries = [summarize_case_document(chunk, max_tokens=1000) for chunk in chunk_text]

    # Construct the final summary prompt
    final_summary_prompt = f"""
    You are looking at the chunked summaries of multiple documents that are all related. 
    Combine the following summaries of the document from different truthful sources into a coherent overall summary:

    <chunked_summaries>
    {"".join(chunk_summaries)}
    </chunked_summaries>

    Focus on these key aspects:
    {details_to_extract_str}

    Provide the summary in bullet points nested within the XML header for each section. For example:
    <case_summary>
        <case_name>
            - [Official Case Name]
        </case_name>
        <presiding_judge>
            - Name: [Judge’s Full Name]
        </presiding_judge>
        <case_summary>
            - Summary: [Brief description of the case]
        </case_summary>
        <key_details>
            - Facts: [Essential facts of the case]
            - Arguments: [Key arguments presented by both sides]
            - Legal Points: [Relevant laws, precedents, or statutes cited]
        </key_details>
        <rationale>
            - Reasoning: [Legal reasoning behind the verdict]
            - Principles Applied: [Judicial principles or frameworks considered]
        </rationale>
        <verdict>
            - Final Judgment: [Outcome of the case]
            - Justification: [Why this verdict was reached]
        </verdict>
    </case_summary>

    If any information is not explicitly stated in the document, note it as "Not specified". Do not include unnecessary preambles.
    """
    
    response = client_atp.messages.create(
        model=model,
        max_tokens=max_tokens,
        system="You are a legal analyst specialising in case law, known for producing highly accurate and structured summaries of legal precedents.",
        messages=[
            {"role": "user", "content": final_summary_prompt},
            {"role": "assistant", "content": "Here is the summary of the legal case document: <summary>"}
        ],
        stop_sequences=["</summary>"]
    )
    
    return response.content[0].text


def summarization_pipeline(text):
    """Processes text, checks token count, and selects the appropriate summarization function."""
    
    # Step 1: Try to count tokens
    try:
        token_count = preprocess_and_count_tokens(text)
        
        # If successful, use the short summarization function
        return summarize_case_document(text)
    
    except Exception as e:
        print("Error encountered while counting tokens. Assuming document exceeds 200,000 tokens.")
        
        # If an error occurs, switch to long summarization function
        return summarize_long_document(text)


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

