import streamlit as st
from client_ui.api import get_chat_response
from client_ui.state import set_response

def render_chat_input():
    """Renders the chat input form using st.chat_input."""
    
    # Placeholder varies by language
    lang = st.session_state.get("language", "ID")
    placeholder = "Tanyakan seputar otomotif..." if lang == "ID" else "Ask about automotive..."
    
    # st.chat_input creates a text box fixed to the bottom or inline with chat
    # It returns the user input when submitted
    query = st.chat_input(placeholder=placeholder)
    
    if query:
        # Display the user message immediately (optional, good for chat feel)
        with st.chat_message("user"):
            st.markdown(query)
            
        try:
            with st.spinner("Sedang memproses..." if lang == "ID" else "Processing..."):
                # Call API
                response = get_chat_response(query)
                # Update State
                set_response(response)
                # Rerun to show the answer in the main area
                st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
