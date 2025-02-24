import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def extract_tagged(text: str, key: str, is_last: bool = False) -> str | None:
    """Extract the first or last occurrence of the text between <key> and </key>"""
    pattern = re.compile(rf"<{key}>(.*?)</{key}>", re.DOTALL)

    if is_last:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
        else:
            return None
    else:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        else:
            return None


def extract_answer(text: str) -> str | None:
    """Extract the answer from the text"""
    return extract_tagged(text, "answer")


def extract_reasoning(text: str) -> str | None:
    """Extract the reasoning from the text"""
    return extract_tagged(text, "reasoning")
