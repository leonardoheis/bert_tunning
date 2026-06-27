import re


def clean_text(text: str) -> str:
    text = text.replace("\f", " ").replace("\xa0", " ")
    text = re.sub(r"\|[-: ]+\|[-: |]+", "", text)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"#+ ", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
