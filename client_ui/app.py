import requests
import streamlit as st


API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def call_api(query: str) -> dict:
    res = requests.post(
        API_URL,
        json={
            "query": query,
            "model_id": MODEL_ID,
        },
        timeout=120,
    )
    res.raise_for_status()
    return res.json()


def handle_query(user_input: str) -> None:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.spinner("Memproses jawaban..."):
        try:
            data = call_api(user_input)
        except requests.RequestException as exc:
            st.error(f"Gagal menghubungi backend: {exc}")
            return
        except ValueError:
            st.error("Response backend bukan JSON yang valid.")
            return

    if "answer" not in data:
        st.error("Format response tidak valid: field `answer` tidak ditemukan.")
        return

    answer = data.get("answer", {}) or {}
    id_text = str(answer.get("id", "")).strip()
    en_text = str(answer.get("en", "")).strip()

    if not id_text and not en_text:
        st.error("Format response tidak valid: `answer.id/en` kosong.")
        return

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": id_text or "-",
            "content_en": en_text or "-",
            "sources": data.get("sources", []),
        }
    )


def render_landing() -> None:
    st.title("Automotive Chat Assistant")
    st.caption("Tanyakan hal seputar otomotif. Jawaban ditampilkan dalam Bahasa Indonesia dan English.")

    st.markdown("### Contoh pertanyaan")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("What is an EV?", use_container_width=True):
            handle_query("What is an EV?")
            st.rerun()
        if st.button("Apa itu mesin diesel?", use_container_width=True):
            handle_query("Apa itu mesin diesel?")
            st.rerun()

    with col2:
        if st.button("Berapa kapasitas baterai Wuling Air EV?", use_container_width=True):
            handle_query("Berapa kapasitas baterai Wuling Air EV?")
            st.rerun()
        if st.button("Bandingkan mobil hybrid vs EV", use_container_width=True):
            handle_query("Bandingkan mobil hybrid vs EV")
            st.rerun()


def render_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown("🇮🇩 **Jawaban**")
                st.markdown(msg["content"])
                st.markdown("🇺🇸 **Answer**")
                st.markdown(msg.get("content_en", "-"))

                sources = msg.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for idx, source in enumerate(sources, start=1):
                            sid = source.get("id", f"source-{idx}")
                            score = source.get("score", "-")
                            content = source.get("content", "")
                            st.markdown(f"**{idx}. {sid}** (score: {score})")
                            if content:
                                st.caption(content)
            else:
                st.markdown(msg["content"])


def main() -> None:
    st.set_page_config(page_title="Automotive Chat UI", page_icon="🚗", layout="centered")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        render_landing()

    render_chat()

    user_input = st.chat_input("Tanya tentang otomotif...")
    if user_input:
        handle_query(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
