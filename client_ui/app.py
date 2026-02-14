import streamlit as st
import time
import re
import sys
import os

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import api
except ImportError:
    # Fallback if running directly from client_ui folder vs root
    import client_ui.api as api

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & STATE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Streamlit AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "language" not in st.session_state:
    st.session_state.language = "ID"  # Default

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    st.session_state.language = st.radio(
        "Language / Bahasa",
        options=["ID", "EN"],
        index=0 if st.session_state.language == "ID" else 1,
        horizontal=True
    )
    st.caption("Select your preferred language for the answer.")

# ─────────────────────────────────────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide main menu and footer for clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container spacing */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 5rem;
        max-width: 800px;
    }

    /* LANDING TITLE */
    .landing-title {
        font-family: "Source Sans Pro", sans-serif;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.5rem;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .landing-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* SUGGESTION PILLS (Capsule Buttons) */
    div[data-testid="stButton"] button {
        border-radius: 50px;
        border: 1px solid #E0E0E0;
        background-color: white;
        color: #31333F;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stButton"] button:hover {
        border-color: #FF4B4B;
        color: #FF4B4B;
        background-color: #FFF9F9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    div[data-testid="stButton"] button:active {
        background-color: #FF4B4B;
        color: white;
    }

    /* CHAT MESSAGES */
    .stChatMessage {
        background-color: transparent;
    }
    div[data-testid="stChatMessageContent"] {
        background-color: #F0F2F6;
        border-radius: 15px;
        padding: 1rem;
        color: #31333F;
    }
    div[data-testid="stChatMessageContent"] p {
        margin: 0;
    }
    
    /* DISCLAIMER */
    .disclaimer-btn {
        position: fixed;
        bottom: 10px;
        left: 10px;
        font-size: 0.7rem;
        color: #999;
        text-decoration: none;
        background: none;
        border: none;
        cursor: pointer;
    }

    /* Hiding the "Deploy" button if visible */
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGIC HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_response(full_text: str, lang: str) -> str:
    """
    Extracts the [ID] or [EN] section from the response.
    Format is usually:
    [ID]
    ...
    [EN]
    ...
    """
    # Normalize newlines
    text = full_text.replace("\r\n", "\n")
    
    # Try regex match
    # Match [ID] content until [EN] or end
    id_match = re.search(r'\[ID\](.*?)(?=\[EN\]|$)', text, re.DOTALL | re.IGNORECASE)
    en_match = re.search(r'\[EN\](.*)', text, re.DOTALL | re.IGNORECASE)
    
    id_text = id_match.group(1).strip() if id_match else ""
    en_text = en_match.group(1).strip() if en_match else ""
    
    # Fallback logic if parsing fails
    if not id_text and not en_text:
        return full_text # Return raw if format is totally broken
        
    if lang == "ID":
        return id_text if id_text else "Maaf, respon bahasa Indonesia tidak tersedia."
    else:
        return en_text if en_text else "Sorry, English response is not available."

def handle_input(query: str):
    """
    Process the user input:
    1. Add user message to state
    2. Call Backend API
    3. Parse and add assistant message to state
    """
    if not query:
        return

    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # 2. Assistant Message
    with st.spinner("Thinking..."):
        try:
            # Call API
            response_data = api.get_chat_response(query)
            raw_response = response_data.get("response", "")
            
            # Parse based on selected language
            final_response = parse_response(raw_response, st.session_state.language)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            
        except Exception as e:
            st.error(f"Error communicating with backend: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def render_landing_ui():
    """
    Renders the initial landing page state.
    """
    # Spacing to push content to visually agreeable center-top
    st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
    
    # Title
    st.markdown('<div class="landing-title">Streamlit AI assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-subtitle">This is an open source app template.</div>', unsafe_allow_html=True)
    
    # Suggestion Pills
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚗 What is an EV?", use_container_width=True):
            handle_input("What is an EV?")
            st.rerun()
        if st.button("🔋 Battery capacity?", use_container_width=True):
            handle_input("How do I check battery capacity?")
            st.rerun()
            
    with col2:
        if st.button("🏎️ Compare engines", use_container_width=True):
            handle_input("Compare V6 and V8 engines")
            st.rerun()
        if st.button("🔧 Maintenance tips", use_container_width=True):
            handle_input("Give me some car maintenance tips")
            st.rerun()

    # Legal Disclaimer (Visual only)
    st.markdown(
        '<button class="disclaimer-btn">Legal disclaimer</button>', 
        unsafe_allow_html=True
    )

def render_chat_ui():
    """
    Renders the chat history.
    """
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP FLOW
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Determine State (Landing vs Chat)
    is_chat_mode = len(st.session_state.messages) > 0

    # 2. Render Main Content
    if not is_chat_mode:
        render_landing_ui()
    else:
        render_chat_ui()

    # 3. Chat Input (Always visible at bottom)
    # The 'paper plane' icon is standard for st.chat_input
    if prompt := st.chat_input("Ask a question..."):
        handle_input(prompt)
        st.rerun()

if __name__ == "__main__":
    main()
