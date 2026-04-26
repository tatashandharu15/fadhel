def ensure_bilingual(text: str) -> str:
    if not text:
        return text

    has_id = "[ID]" in text
    has_en = "[EN]" in text

    # Already valid bilingual format
    if has_id and has_en:
        return text

    # If only ID exists, duplicate to EN
    if has_id and not has_en:
        content = text.replace("[ID]", "").strip()
        return f"[ID]\n{content}\n\n[EN]\n{content}"

    # If there is no bilingual marker at all
    clean = text.strip()
    return f"[ID]\n{clean}\n\n[EN]\n{clean}"
