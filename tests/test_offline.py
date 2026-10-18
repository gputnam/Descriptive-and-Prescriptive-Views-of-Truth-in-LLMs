"""Offline tests: grader prompt/parsing/transport (with a fake HTTP session) and the
answer-swap helpers. No network, no GPU.

    pytest tests/
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthsteer import grader  # noqa: E402
from truthsteer.swap import build_prefix, extract_boxed, last_think_box, norm_answer  # noqa: E402

GCFG = {"base_url": "https://api.anthropic.com", "model": "claude-sonnet-5", "temperature": 0.0,
        "max_tokens": 400, "max_retries": 3, "think_tail_chars": 2000}
VERDICT = ('{"final_answer": "120", "verdict": "not_flipped", "matches_original": true, '
           '"matches_gt": true, "reasoning": "Reconfirms 120."}')


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": dict(json)})
        return self.responses.pop(0)


def ok(text, stop="end_turn", thinking=False):
    blocks = ([{"type": "thinking", "thinking": "..."}] if thinking else []) + \
             ([{"type": "text", "text": text}] if text is not None else [])
    return FakeResponse(200, {"content": blocks, "stop_reason": stop, "usage": {"input_tokens": 10}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(grader.time, "sleep", lambda s: None)


def test_user_message_finished_response():
    msg = grader.build_user_message("P?", "120", "120", "x" * 3000 + "</think>\nThe answer is \\boxed{120}.")
    assert "MODEL'S ORIGINAL FINAL ANSWER: 120" in msg
    assert "x" * 2000 in msg and "x" * 2001 not in msg    # only the think tail is shown
    assert msg.endswith("The answer is \\boxed{120}.")


def test_user_message_cut_off_response():
    msg = grader.build_user_message("P?", "1", "2", "Wait, let me recheck " * 50)
    assert "cut off before it finished" in msg


@pytest.mark.parametrize("text", [VERDICT, f"```json\n{VERDICT}\n```", f"Here you go:\n{VERDICT}\nDone."])
def test_extract_json_variants(text):
    assert grader.extract_json(text)["verdict"] == "not_flipped"


def test_parse_reply_skips_thinking_block():
    out = grader.parse_reply(ok(VERDICT, thinking=True).json())
    assert out["matches_gt"] is True and out["_usage"] == {"input_tokens": 10}


def test_parse_reply_rejects_bad_verdict():
    with pytest.raises(ValueError):
        grader.parse_reply(ok(VERDICT.replace("not_flipped", "firm")).json())


def test_grade_one_retries_then_succeeds():
    sess = FakeSession([FakeResponse(429, {}), ok(VERDICT)])
    out = grader.grade_one(sess, "https://api.anthropic.com", {}, GCFG, "P?", "120", "120", "...")
    assert out["verdict"] == "not_flipped" and len(sess.calls) == 2
    assert sess.calls[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert sess.calls[0]["payload"]["system"] == grader.SYSTEM


def test_grade_one_raises_budget_when_thinking_eats_it():
    sess = FakeSession([ok(None, stop="max_tokens", thinking=True), ok(VERDICT)])
    grader.grade_one(sess, "b", {}, GCFG, "P?", "1", "1", "...")
    assert sess.calls[1]["payload"]["max_tokens"] == 1600


def test_grade_one_gives_up():
    sess = FakeSession([FakeResponse(500, {})] * 3)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        grader.grade_one(sess, "b", {}, GCFG, "P?", "1", "1", "...")


def test_grade_one_client_error_is_not_retried_forever():
    sess = FakeSession([FakeResponse(401, {"error": "bad key"})] * 3)
    with pytest.raises(RuntimeError):
        grader.grade_one(sess, "b", {}, GCFG, "P?", "1", "1", "...")


def test_resolve_auth(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        grader.resolve_auth(GCFG)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    base, h = grader.resolve_auth(GCFG)
    assert base == "https://api.anthropic.com" and h["x-api-key"] == "k"
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example/")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")
    base, h = grader.resolve_auth(GCFG)
    assert base == "https://proxy.example" and h["Authorization"] == "Bearer t"


COT = ("Solve. <think>\nSo it is 120.\n**Final Answer**\nThe answer is $\\boxed{\\dfrac{1}{2}}$\n"
       "</think>\n\\boxed{\\dfrac{1}{2}}")


def test_swap_helpers():
    s, e = last_think_box(COT)
    assert COT[s:e] == "\\dfrac{1}{2}"
    prefix, common, box = build_prefix(COT, "384")
    assert prefix == common + "384}" and box == "\\dfrac{1}{2}" and "</think>" not in prefix
    assert norm_answer(" $\\dfrac{1}{2}$ ") == norm_answer("\\frac{1}{2}")
    assert extract_boxed("so \\boxed{\\frac{a}{b}} and \\boxed{7}.") == "7"
    assert last_think_box("no think block \\boxed{1}") is None
