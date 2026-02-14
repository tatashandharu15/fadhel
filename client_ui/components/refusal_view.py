import streamlit as st

def is_refusal(text: str) -> bool:
    """
    Heuristic to detect if the response is a refusal.
    Based on backend refusal templates.
    """
    if not text:
        return False
        
    # Check for standard refusal phrases defined in backend strategies
    refusal_keywords = [
        "hanya mendukung pertanyaan seputar otomotif",
        "only supports automotive-related questions",
        "data otomotif yang relevan tidak tersedia",
        "relevant automotive data is not available"
    ]
    
    # Check if any keyword exists in the text
    for keyword in refusal_keywords:
        if keyword.lower() in text.lower():
            return True
            
    return False

def render_refusal_warning(language="ID"):
    """Renders a warning box for refused queries."""
    
    title = "Permintaan Ditolak / Refused"
    message = "Sistem tidak dapat menjawab pertanyaan ini karena berada di luar lingkup otomotif atau data tidak tersedia."
    
    if language == "EN":
        title = "Request Refused"
        message = "The system cannot answer this question because it is outside the automotive scope or data is unavailable."
    elif language == "ID":
        title = "Permintaan Ditolak"
        message = "Sistem tidak dapat menjawab pertanyaan ini karena berada di luar lingkup otomotif atau data tidak tersedia."

    st.markdown(
        f"""
        <div class="refusal-box">
            <span>⚠️</span>
            <div>
                <strong>{title}</strong><br>
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
