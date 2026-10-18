"""Answer-swap construction for the Reasoned Truth data.

Take a chain of thought that reasoned to the CORRECT answer and wrote it as
`\\boxed{X}` at the end of its <think> block. Cut the text right after that box and
put a different answer inside it. Only the LAST box inside <think> is touched;
earlier sentences that state the correct answer are left as they are.
"""
import re

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def last_think_box(full_cot: str):
    """(start, end) character span of the CONTENT of the last \\boxed{...} inside
    <think>...</think>, or None. Braces are matched, so \\boxed{\\dfrac{1}{2}} works."""
    if THINK_OPEN not in full_cot or THINK_CLOSE not in full_cot:
        return None
    a = full_cot.index(THINK_OPEN)
    b = full_cot.index(THINK_CLOSE)
    i = full_cot.rfind("\\boxed{", a, b)
    if i < 0:
        return None
    start = i + len("\\boxed{")
    depth, j = 1, start
    while j < b:
        ch = full_cot[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, j
        j += 1
    return None   # unbalanced inside <think>


def norm_answer(s: str) -> str:
    """Loose normal form, used only to check that the box holds the final answer and
    that a 'wrong' answer is not trivially the same string."""
    s = s.strip().strip("$").strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s


def build_prefix(full_cot: str, answer: str):
    """(prefix, common, box_content): the CoT cut right after its last think-box with
    `answer` inside; `common` is everything through '\\boxed{'. None if no usable box."""
    span = last_think_box(full_cot)
    if span is None:
        return None
    s, e = span
    common = full_cot[:s]
    return common + answer + "}", common, full_cot[s:e]


def extract_boxed(solution: str):
    """Content of the LAST \\boxed{...} in a MATH reference solution (nested braces ok)."""
    idx = solution.rfind("\\boxed")
    if idx == -1:
        return None
    i = solution.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(solution)):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[i + 1:j].strip()
    return None
