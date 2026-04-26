import streamlit as st


def render_answer(response_data: dict, language: str = "id", show_both: bool = False):
    """Renders answer from structured backend contract."""
    if not response_data or "answer" not in response_data:
        return

    answer = response_data.get("answer", {}) or {}
    id_text = str(answer.get("id", "")).strip()
    en_text = str(answer.get("en", "")).strip()

    if show_both:
        st.markdown("🇮🇩 " + id_text)
        st.markdown("🇬🇧 " + en_text)
        return

    if language.lower() == "id":
        st.markdown(
            f"""
            <div class="answer-section answer-id">
                <div class="answer-header">🇮🇩 Jawaban</div>
                <div>{id_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="answer-section answer-en">
            <div class="answer-header">🇬🇧 Answer</div>
            <div>{en_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
