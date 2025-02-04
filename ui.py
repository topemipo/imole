import requests
import streamlit as st

# 🚀 Replace with your actual FastAPI server URL
API_URL = "https://imole-glmu8.ondigitalocean.app/chat/"

# Streamlit UI Setup
st.set_page_config(page_title="⚖️ Legal AI Chatbot", layout="wide")
st.title("⚖️ Legal AI Chatbot")
st.markdown("Ask any legal question and receive AI-generated responses based on legal case laws.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for chat in st.session_state["chat_history"]:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User input field
user_query = st.chat_input("Ask your legal question...")

if user_query:
    # Add user input to chat history
    st.session_state["chat_history"].append({"role": "user", "content": user_query})

    # 🔗 Send request to FastAPI backend
    response = requests.post(API_URL, json={"user_query": user_query})

    if response.status_code == 200:
        ai_response = response.json().get("response", "⚠️ No response received.")
    else:
        ai_response = "⚠️ Error: Could not connect to the backend."

    # Add AI response to chat history
    st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})

    # Display AI response in chat format
    with st.chat_message("assistant"):
        st.markdown(ai_response)