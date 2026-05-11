import streamlit as st


def _to_html_with_breaks(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def render_answer(response_data: dict, language: str = "id", show_both: bool = False):
    """Renders answer from structured backend contract."""
    if not response_data or "answer" not in response_data:
        return

    answer = response_data.get("answer", {}) or {}
    id_text = str(answer.get("id", "")).strip()
    en_text = str(answer.get("en", "")).strip()
    id_html = _to_html_with_breaks(id_text)
    en_html = _to_html_with_breaks(en_text)

    if show_both:
        st.markdown("ID " + id_html, unsafe_allow_html=True)
        st.markdown("EN " + en_html, unsafe_allow_html=True)
        return

    if language.lower() == "id":
        st.markdown(
            f"""
            <div class="answer-section answer-id">
                <div class="answer-header">Jawaban</div>
                <div>{id_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="answer-section answer-en">
            <div class="answer-header">Answer</div>
            <div>{en_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )