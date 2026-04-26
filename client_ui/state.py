import streamlit as st

def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "current_response" not in st.session_state:
        st.session_state.current_response = None
        
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
        
    if "language" not in st.session_state:
        st.session_state.language = "id"  # Default to Indonesian

def set_response(response_data):
    """Update the current response in session state."""
    st.session_state.current_response = response_data

def get_response():
    """Get the current response from session state."""
    return st.session_state.current_response
