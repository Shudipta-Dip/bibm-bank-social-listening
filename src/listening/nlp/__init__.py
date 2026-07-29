"""NLP package."""

from listening.nlp.language import detect_language, enrich_languages
from listening.nlp.sentiment import enrich_sentiment
from listening.nlp.themes import enrich_themes

__all__ = [
    "detect_language",
    "enrich_languages",
    "enrich_sentiment",
    "enrich_themes",
]
