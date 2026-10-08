"""Keyword search (BM25) over the Larkspur knowledge base."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from helpdesk.store import KB_DIR

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    }
)


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    tags: str
    body: str


_SUFFIXES = ("ations", "ation", "ing", "ed", "es", "s")


def _stem(word: str) -> str:
    # Crude suffix folding so "leaking", "installed" and "refunds" find "leak", "installation", "refund".
    for suffix in _SUFFIXES:
        if (
            word.endswith(suffix)
            and len(word) - len(suffix) >= 3
            and not word.endswith("ss")
        ):
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in _STOP]


def load_articles(kb_dir: Path = KB_DIR) -> list[Article]:
    articles = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text()
        _, front, body = text.split("---", 2)
        meta = dict(line.split(": ", 1) for line in front.strip().splitlines())
        articles.append(
            Article(meta["id"], meta["title"], meta.get("tags", ""), body.strip())
        )
    return articles


def search(
    query: str, top_k: int = 3, kb_dir: Path = KB_DIR
) -> list[tuple[Article, float]]:
    articles = load_articles(kb_dir)
    # Title and tags count three times, so an article about the topic outranks one that mentions it.
    docs = [_tokens(f"{a.title} {a.tags} " * 3 + a.body) for a in articles]
    avg = sum(len(d) for d in docs) / len(docs)
    df = Counter(term for d in docs for term in set(d))
    terms = _tokens(query)
    scored = []
    for article, doc in zip(articles, docs):
        tf = Counter(doc)
        score = 0.0
        for term in terms:
            if term not in tf:
                continue
            idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            score += (
                idf * tf[term] * 2.2 / (tf[term] + 1.2 * (0.25 + 0.75 * len(doc) / avg))
            )
        if score > 0:
            scored.append((article, round(score, 2)))
    scored.sort(key=lambda pair: -pair[1])
    return scored[:top_k]


def render(results: list[tuple[Article, float]]) -> str:
    if not results:
        return "No knowledge-base articles matched. Try different keywords."
    return "\n\n".join(
        f"=== {a.id}: {a.title} (score {s}) ===\n{a.body}" for a, s in results
    )
