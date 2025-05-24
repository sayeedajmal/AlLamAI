import streamlit as st
import requests

st.set_page_config(page_title="ALLaM Chat", layout="centered")

st.markdown("<h1 style='text-align: center;'>🤖 ALLaM Chat</h1>", unsafe_allow_html=True)

# Session-based chat history (clears on reload)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Update this if your API is hosted differently or on a remote server
API_URL = "http://127.0.0.1:8000/generate"

def generate_reply(prompt):
    try:
        response = requests.post(API_URL, json={"prompt": prompt, "max_new_tokens": 200})
        response.raise_for_status()
        return response.json().get("response", "No response received.")
    except Exception as e:
        return f"Error: {e}"

# Chat box UI
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Ask me anything...", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("Thinking..."):
            reply = generate_reply(user_input)
        st.session_state.chat_history.append(("bot", reply))

# Display chat history with simple styling
for sender, msg in st.session_state.chat_history:
    align = "flex-end" if sender == "user" else "flex-start"
    bubble_color = "#DCF8C6" if sender == "user" else "#F1F0F0"
    st.markdown(
        f"""
        <div style='display: flex; justify-content: {align}; padding: 4px;'>
            <div style='background-color: {bubble_color}; padding: 10px 15px; border-radius: 12px; max-width: 80%;'>
                {msg}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
