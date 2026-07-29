"""Minimal unit checks for schema, language, themes, HITL sampling."""

from __future__ import annotations

from datetime import datetime, timezone

from listening.hitl import agreement_metrics, stratified_sample
from listening.nlp.language import detect_language
from listening.nlp.themes import tag_themes
from listening.normalize.schema import build_mention, rating_to_sentiment
from listening.normalize import from_google_play


def test_rating_sentiment():
    assert rating_to_sentiment(5) == "positive"
    assert rating_to_sentiment(3) == "neutral"
    assert rating_to_sentiment(1) == "negative"


def test_detect_language():
    assert detect_language("This app is great and fast") == "en"
    assert detect_language("অ্যাপটা খুব ভালো") == "bn"
    assert detect_language("app ta khub bhalo lagche") == "mixed"


def test_themes():
    themes = tag_themes("Login OTP crash slow app", {"app_ux": ["login", "otp", "crash"], "fees": ["fee"]})
    assert "app_ux" in themes
    assert "fees" not in themes


def test_google_play_normalize():
    since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    raw = {
        "reviewId": "abc123",
        "userName": "Tester",
        "content": "Good app",
        "score": 5,
        "at": "2025-01-01T00:00:00+00:00",
        "thumbsUpCount": 2,
        "replyContent": "Thanks",
        "repliedAt": "2025-01-02T00:00:00+00:00",
    }
    rows = from_google_play(raw, "brac_bank", since)
    assert len(rows) == 2
    assert rows[0].content_type == "review"
    assert rows[1].content_type == "review_reply"
    assert rows[0].record_id


def test_stratified_and_agreement():
    records = [
        {"brand": "brac_bank", "source": "google_play", "language": "en", "record_id": f"r{i}", "text": "x"}
        for i in range(10)
    ] + [
        {"brand": "scb_bangladesh", "source": "app_store", "language": "bn", "record_id": f"s{i}", "text": "y"}
        for i in range(10)
    ]
    sample = stratified_sample(records, n=6)
    assert len(sample) == 6
    gold = [
        {"human_sentiment_label": "positive", "model_sentiment_label": "positive"},
        {"human_sentiment_label": "negative", "model_sentiment_label": "negative"},
        {"human_sentiment_label": "neutral", "model_sentiment_label": "positive"},
    ]
    m = agreement_metrics(gold)
    assert m["n_labeled"] == 3
    assert m["accuracy"] is not None


def test_build_mention_hash():
    m = build_mention(
        brand="brac_bank",
        source="google_play",
        content_type="review",
        native_id="n1",
        text="hello",
        author_key="userA",
        rating=4,
    )
    assert m.author_hash and len(m.author_hash) == 64
    assert m.rating_sentiment == "positive"


if __name__ == "__main__":
    test_rating_sentiment()
    test_detect_language()
    test_themes()
    test_google_play_normalize()
    test_stratified_and_agreement()
    test_build_mention_hash()
    print("all tests passed")
