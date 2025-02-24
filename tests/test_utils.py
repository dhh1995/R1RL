from r1rl.utils import extract_answer, extract_reasoning, extract_tagged


def test_extract_tagged():
    text = "<xxx>42</xxx>"
    assert extract_tagged(text, "xxx") == "42"
    text = "<xxx>\n42\n</xxx> <xxx>43</xxx>"
    assert extract_tagged(text, "xxx") == "42"
    assert extract_tagged(text, "xxx", is_last=True) == "43"


def test_extract_reasoning():
    text = "<reasoning>The reasoning is 42</reasoning>."
    assert extract_reasoning(text) == "The reasoning is 42"


def test_extract_answer():
    text = "The answer is <answer>42</answer>."
    assert extract_answer(text) == "42"
