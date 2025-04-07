import streamlit as st
from main import graph  # ✅ Import the compiled LangGraph

# Streamlit UI setup
st.set_page_config(page_title="LangGraph Demo", page_icon="🌦️")
st.title("LangGraph Chatbot 🌤️📄")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input box
user_input = st.text_input("Ask a question (weather or PDF-related):", "")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    # Process input through LangGraph pipeline
    try:
        state = {"input": user_input}
        result = graph.invoke(state)["response"]
    except Exception as e:
        result = f"Error: {str(e)}"

    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "text": result})

# Display chat history
for chat in st.session_state.chat_history:
    role = "👤" if chat["role"] == "user" else "🤖"
    st.markdown(f"{role} **{chat['text']}**")
