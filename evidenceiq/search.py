from __future__ import annotations

import math
import re
from collections import Counter

from evidenceiq.models import EvidenceItem, SearchResult

TOKEN_RE = re.compile(r"[a-zA-Z0-9$@._-]+")
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "did",
    "evidence",
    "for",
    "from",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
    "without",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


class EvidenceSearch:
    def __init__(self, items: list[EvidenceItem]):
        self.items = items
        self.documents = [tokenize(" ".join([i.title, i.body, " ".join(i.entities.get("risk_terms", []))])) for i in items]
        self.doc_freq = Counter(token for doc in self.documents for token in set(doc))

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scored: list[SearchResult] = []
        query_vec = Counter(query_terms)
        for item, tokens in zip(self.items, self.documents):
            score = self._cosine(query_vec, Counter(tokens))
            exact = tuple(sorted(set(query_terms).intersection(tokens)))
            if score >= 0.08 or exact:
                excerpt = best_excerpt(item.body, query_terms)
                scored.append(SearchResult(item, score + len(exact) * 0.05, exact, excerpt))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:limit]

    def _weight(self, token: str, count: int) -> float:
        idf = math.log((1 + len(self.items)) / (1 + self.doc_freq[token])) + 1
        return count * idf

    def _cosine(self, left: Counter, right: Counter) -> float:
        keys = set(left) | set(right)
        left_vec = {key: self._weight(key, left[key]) for key in keys}
        right_vec = {key: self._weight(key, right[key]) for key in keys}
        dot = sum(left_vec[key] * right_vec[key] for key in keys)
        left_norm = math.sqrt(sum(value * value for value in left_vec.values()))
        right_norm = math.sqrt(sum(value * value for value in right_vec.values()))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)


def best_excerpt(text: str, terms: list[str], window: int = 280) -> str:
    lower = text.lower()
    positions = [lower.find(term.lower()) for term in terms if lower.find(term.lower()) >= 0]
    start = min(positions) if positions else 0
    start = max(0, start - 80)
    excerpt = text[start : start + window]
    return " ".join(excerpt.split())
