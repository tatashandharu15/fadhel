import streamlit as st

def render_sources(sources: list):
    """Renders the sources list."""
    if not sources:
        return
        
    with st.expander("📚 Sumber Data / Sources", expanded=False):
        for source in sources:
            score = source.get("score", 0.0)
            title = source.get("title", "Unknown Source")
            
            st.markdown(
                f"""
                <div class="source-box">
                    <div class="source-header">
                        <span>📄 {title}</span>
                        <span class="source-score">Sim: {score:.2f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
