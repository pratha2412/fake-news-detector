import re


def preprocess_text(text: str) -> str:
    """Shared text cleaning for training and inference."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(
        r'(?i)^(From|Subject|Organization|Lines|NNTP-Posting-Host):.*$',
        '',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
