from textwrap import dedent
from typing import Dict, List

from termcolor import colored

REASONING_TAG = "reasoning"
ANSWER_TAG = "answer"
COLOR_MAP = {
    "system": "yellow",
    "user": "cyan",
}


def display_prompt(prompt: List[Dict[str, str]] | str) -> str:
    """Display the prompt in a readable format"""
    if isinstance(prompt, str):
        return colored(prompt, "cyan")
    message = ""
    for p in prompt:
        message += colored(f"Role({p['role']}): {p['content']}\n", COLOR_MAP[p["role"]])
    return message
