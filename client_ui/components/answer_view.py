import streamlit as st
import re
from client_ui.components.refusal_view import is_refusal, render_refusal_warning

def parse_bilingual_response(text: str):
    """
    Splits the response into ID and EN sections.
    Expected format: [ID] ... [EN] ...
    """
    id_content = ""
    en_content = ""
    
    # Try to split by [EN] first
    parts = text.split("[EN]")
    
    if len(parts) >= 2:
        # Part 0 contains [ID] content
        raw_id = parts[0]
        # Part 1 contains [EN] content
        en_content = parts[1].strip()
        
        # Remove [ID] marker from raw_id
        id_content = raw_id.replace("[ID]", "").strip()
    else:
        # Fallback if format is broken (should not happen with strict prompts)
        id_content = text
        en_content = ""
        
    return id_content, en_content

def render_answer(response_data: dict, language: str = "ID"):
    """Renders the bilingual answer."""
    if not response_data or "response" not in response_data:
        return
        
    raw_text = response_data["response"]
    
    # Check for refusal
    if is_refusal(raw_text):
        render_refusal_warning(language)
        # Even if refused, we might want to stop here or show partial text?
        # Usually refusal is just the warning.
        return
    
    # Parse Bilingual
    id_text, en_text = parse_bilingual_response(raw_text)
    
    # Render based on selected language
    if language == "ID":
        st.markdown(
            f"""
            <div class="answer-section answer-id">
                <div class="answer-header">🇮🇩 Jawaban</div>
                <div>{id_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif language == "EN":
        st.markdown(
            f"""
            <div class="answer-section answer-en">
                <div class="answer-header">🇬🇧 Answer</div>
                <div>{en_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Fallback: Show Both
        st.markdown(
            f"""
            <div class="answer-section answer-id">
                <div class="answer-header">🇮🇩 Jawaban (ID)</div>
                <div>{id_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="answer-section answer-en">
                <div class="answer-header">🇬🇧 Answer (EN)</div>
                <div>{en_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
