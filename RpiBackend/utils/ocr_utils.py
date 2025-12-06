def clean_and_format_text(text):
    if not text or len(text.strip()) < 3:
        return None
    
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = " ".join(lines)

    txt = cleaned.lower()

    # Example smart meanings
    if "speed" in txt:
        return f"Speed-related sign detected: {cleaned}"
    if "private" in txt or "no entry" in txt:
        return f"Restricted area sign detected: {cleaned}"
    if "danger" in txt:
        return f"Warning sign detected: {cleaned}"

    return cleaned
