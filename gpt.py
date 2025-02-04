import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from query_db_functions import (
    augment_query_generated,
    query_chroma_collection,
    get_file_contents,
    summarize_text_with_map_reduce,
    generate_response,
)

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# ChromaDB connection
db_path = os.getenv("DATABASE_PATH")
db_client = chromadb.PersistentClient(path=db_path)
folder_path = os.path.join(os.getenv("CASEDOCS_PATH"), "*.txt")

# Initialise session state for chat history if it doesn't exist yet
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Legal AI Chatbot", layout="wide")
st.title("⚖️ Legal AI Chatbot")
st.markdown("Ask any legal question and receive AI-generated responses based on legal case laws.")

# Display existing chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User input field for query
user_query = st.chat_input("Ask your legal question...")

if user_query:
    # Immediately record and display the user query
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Use a dedicated assistant chat message block for the assistant's response.
    with st.chat_message("assistant"):
        # Use a spinner to indicate processing.
        with st.spinner("Processing your query..."):
            # A short delay to ensure the UI updates with the spinner
            time.sleep(0.5)
            augmented_query = augment_query_generated(user_query)
            final_query = f"{user_query} {augmented_query}"

            embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-ada-002"
            )

            legal_case_collection = db_client.get_collection(
                name="case-docs-collection",
                embedding_function=embedding_function
            )

            retrieved_chunks = query_chroma_collection(legal_case_collection, final_query, n_results=1)
            if not retrieved_chunks:
                ai_response = "⚠️ No relevant legal documents found. Consider consulting a lawyer."
            else:
                case_doc = retrieved_chunks[0]["source"]
                file_content = get_file_contents(folder_path, case_doc)
                summary = summarize_text_with_map_reduce(file_content, max_length=1000)
                ai_response = generate_response(user_query, summary)

        # After processing, simulate a streaming update of the assistant's response.
        response_placeholder = st.empty()
        full_response = ""
        for char in ai_response:
            full_response += char
            response_placeholder.markdown(full_response)
            time.sleep(0.02)

    # Append the complete assistant response to the session state.
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})