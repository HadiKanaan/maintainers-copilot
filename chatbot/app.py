# Purpose: Streamlit UI with authentication and chat interface.
# Significance: Minimal frontend for the Maintainer's Copilot.
import streamlit as st
import requests

API_URL = st.secrets.get("API_URL", "http://localhost:8000")


# Render login form and store JWT on success.
def _login() -> None:
    """Render login form and store JWT on success."""
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        r = requests.post(f"{API_URL}/auth/login", json={"email": email, "password": password})
        if r.status_code == 200:
            st.session_state.jwt = r.json()["access_token"]
            st.session_state.view = "chat"
        else:
            st.error("Login failed")


# Fetch conversations for sidebar list.
def _fetch_conversations() -> list:
    """Fetch conversations for sidebar list."""
    r = requests.get(f"{API_URL}/chat/conversations", headers={"Authorization": f"Bearer {st.session_state.jwt}"})
    if r.status_code == 200:
        return r.json()
    return []


# Render chat interface with history and input box.
def _chat_ui() -> None:
    """Render chat interface with history and input box."""
    st.header("Maintainer's Copilot")

    if st.sidebar.button("New conversation"):
        r = requests.post(f"{API_URL}/chat/conversations", headers={"Authorization": f"Bearer {st.session_state.jwt}"})
        if r.status_code == 200:
            st.session_state.conversation_id = r.json()["id"]
            st.session_state.messages = []

    conversations = _fetch_conversations()
    for conv in conversations:
        if st.sidebar.button(f"Conversation {conv['id']}"):
            st.session_state.conversation_id = conv["id"]
            st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
            st.caption(f"trace_id: {msg.get('trace_id', '')}")

    prompt = st.chat_input("Type your message")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        payload = {"conversation_id": str(st.session_state.conversation_id), "message": prompt}
        r = requests.post(
            f"{API_URL}/chat/message",
            json=payload,
            headers={"Authorization": f"Bearer {st.session_state.jwt}"},
        )
        if r.status_code == 200:
            resp = r.json()
            st.session_state.messages.append({"role": "assistant", "content": resp["response"], "trace_id": resp["trace_id"]})
        else:
            st.error("Chat request failed")

    if st.button("Logout"):
        st.session_state.clear()


if "view" not in st.session_state:
    st.session_state.view = "login"

if "jwt" not in st.session_state:
    st.session_state.jwt = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = ""

if st.session_state.jwt:
    _chat_ui()
else:
    _login()
