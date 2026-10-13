"""Run: pytest -q   (no API key needed; the Claude path is exercised with a stub)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app import dating
from app.analyzer import analyze
from app.personas import PERSONAS


def people():
    return [dict(p, analysis=analyze(p)) for p in PERSONAS]


def test_25_personas_all_labelled_fictional():
    assert len(PERSONAS) == 25
    assert all(p["fictional"] for p in PERSONAS)


def test_every_extracted_item_has_evidence_from_the_two_sources():
    for p in PERSONAS:
        a = analyze(p)
        assert a["hobbies"] and a["values"] and a["needs"]
        for item in a["hobbies"] + a["values"]:
            for e in item["evidence"]:
                assert e["source"].startswith(("LinkedIn", "Instagram"))


def test_date_is_order_independent_and_scores_in_range():
    ps = people()
    d1 = dating.run_date(ps[0], ps[1])
    d2 = dating.run_date(ps[1], ps[0])
    assert d1 == d2
    assert 0 <= d1["scores"]["a_view"] <= 100
    assert d1["turns"][-1]["kind"] == "verdict" and d1["turns"][-2]["kind"] == "verdict"


def test_deal_breaker_clash_is_penalised():
    ps = {p["id"]: p for p in people()}
    d = dating.run_date(ps["vikram-rao"], ps["karan-bhatia"])
    assert (d["a"], d["b"]) == ("karan-bhatia", "vikram-rao")          # ids are ordered
    assert d["breakdown"]["penalty_b"] == dating.PENALTY               # Vikram wrote "Deal-breaker: smoking"
    assert d["breakdown"]["penalty_a"] == 0                            # Karan has no stated deal-breaker
    assert any("smoking" in w for w in d["reasons"]["b"]["watch"])


def test_rankings_cover_everyone_else_sorted():
    ps = people()
    dates = dating.run_all(ps)
    assert len(dates) == 300
    r = dating.rank_for("kabir-nair", ps, dates)
    assert len(r) == 24 and [x["score"] for x in r] == sorted((x["score"] for x in r), reverse=True)


def test_claude_path_uses_separate_agent_calls(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stub")
    calls = []

    def fake(system, messages, max_tokens=300):
        calls.append(system[:30])
        return '{"score": 70, "reason": "ok", "meet_again": "yes"}' if "Return ONLY JSON" in messages[0]["content"] else "hello"
    monkeypatch.setattr(dating, "llm_call", fake)
    ps = people()
    d = dating.run_date(ps[0], ps[1], "claude")
    assert d["engine"] == "claude" and len(calls) == 8


def test_api_requires_consent_and_pasted_text():
    from app.main import app
    c = TestClient(app)
    base = {"name": "T", "linkedin_url": "https://www.linkedin.com/in/t", "instagram_url": "https://www.instagram.com/t"}
    assert c.post("/api/people", json={**base, "consent": False}).status_code == 422
    assert c.post("/api/people", json={**base, "consent": True}).status_code == 422          # no text pasted
    assert c.post("/api/people", json={**base, "consent": True, "linkedin_url": "https://evil.com/x",
                                       "linkedin_text": "x" * 50, "instagram_text": "y" * 50}).status_code == 422
