"""
Fetches real Harry Potter content from two sources:
  1. HP Fandom Wiki (harrypotter.fandom.com) — HP-specific detail
  2. Wikipedia MediaWiki API (en.wikipedia.org) — broad coverage fallback

Both use the standard MediaWiki API (no scraping, no auth required).
"""

from __future__ import annotations
import json
import os
import re
import time

import requests

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wiki_cache.json")

FANDOM_API  = "https://harrypotter.fandom.com/w/api.php"   # note /w/ path
WIKI_API    = "https://en.wikipedia.org/w/api.php"

HP_ARTICLES = [
    # Characters
    "Harry Potter (character)", "Hermione Granger", "Ron Weasley",
    "Albus Dumbledore", "Lord Voldemort", "Severus Snape",
    "Draco Malfoy", "Neville Longbottom", "Luna Lovegood",
    "Sirius Black", "Remus Lupin", "Bellatrix Lestrange",
    "Ginny Weasley", "Rubeus Hagrid", "Minerva McGonagall",
    "Cedric Diggory", "Peter Pettigrew", "Dobby (character)",
    "James Potter", "Lily J. Potter",
    # Locations
    "Hogwarts", "Diagon Alley", "Hogsmeade",
    "Azkaban", "Ministry of Magic (Harry Potter)", "Forbidden Forest (Harry Potter)",
    "Gringotts", "Godric's Hollow",
    "12 Grimmauld Place", "Room of Requirement",
    # Spells
    "Expelliarmus", "Avada Kedavra", "Expecto Patronum",
    "Wingardium Leviosa", "Alohomora", "Stupefy (spell)",
    "Sectumsempra", "Cruciatus Curse", "Legilimency",
    # Objects / Artifacts
    "Elder Wand", "Invisibility cloak (Harry Potter)", "Resurrection Stone",
    "Philosopher's Stone (Harry Potter)", "Time-Turner", "Marauder's Map",
    "Horcrux", "Deathly Hallows (objects)",
    # Events / Groups
    "Battle of Hogwarts", "Triwizard Tournament",
    "Order of the Phoenix (organisation)", "Death Eaters",
    "Quidditch", "Gryffindor", "Slytherin",
    "Harry Potter and the Philosopher's Stone",
    "Harry Potter and the Deathly Hallows",
]

WIKI_FALLBACK_TITLES = {
    "Harry Potter (character)": "Harry Potter",
    "Dobby (character)": "Dobby (Harry Potter)",
    "Ministry of Magic (Harry Potter)": "Ministry of Magic",
    "Forbidden Forest (Harry Potter)": "Forbidden Forest",
    "Stupefy (spell)": "Stupefy",
    "Invisibility cloak (Harry Potter)": "Invisibility cloak",
    "Philosopher's Stone (Harry Potter)": "Philosopher's stone",
    "Deathly Hallows (objects)": "Deathly Hallows",
    "Order of the Phoenix (organisation)": "Order of the Phoenix",
}

MAX_CHARS = 6000


def _mediawiki_fetch(api_url: str, title: str, source_label: str) -> dict | None:
    """Fetch plain-text article extract via any MediaWiki API endpoint."""
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "format": "json",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
    }
    try:
        resp = requests.get(api_url, params=params, timeout=12,
                            headers={"User-Agent": "HPKnowledgeGraph/1.0 (aneelaveldi09@gmail.com)"})
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        extract = page.get("extract", "").strip()
        if not extract or len(extract) < 150:
            return None
        return {
            "title": page.get("title", title),
            "source": source_label,
            "text": _clean(extract)[:MAX_CHARS],
        }
    except Exception as e:
        return None


def _clean(text: str) -> str:
    text = re.sub(r"==+[^=]+=+", " ", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_corpus(use_cache: bool = True) -> list[dict]:
    if use_cache and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            corpus = json.load(f)
        print(f"  Loaded {len(corpus)} articles from cache")
        return corpus

    print(f"Fetching {len(HP_ARTICLES)} HP articles from Wikipedia & HP Fandom Wiki…")
    corpus = []
    for title in HP_ARTICLES:
        print(f"  → {title}", end=" ", flush=True)

        # 1. Try HP Fandom wiki first
        article = _mediawiki_fetch(FANDOM_API, title, "hp_fandom_wiki")

        # 2. Fall back to Wikipedia
        if article is None:
            wiki_title = WIKI_FALLBACK_TITLES.get(title, title)
            article = _mediawiki_fetch(WIKI_API, wiki_title, "wikipedia")

        if article:
            # Tag with the lookup title for display
            article["query_title"] = title
            corpus.append(article)
            print(f"✓ [{article['source']}] {len(article['text'])} chars")
        else:
            print("✗ (skipped)")

        time.sleep(0.2)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"\nFetched {len(corpus)}/{len(HP_ARTICLES)} articles — cached to {CACHE_PATH}")
    return corpus


def corpus_to_sentences(corpus: list[dict], min_len: int = 30) -> list[dict]:
    """Split each article into sentences with source metadata."""
    sentences = []
    for article in corpus:
        for sent in re.split(r'(?<=[.!?])\s+', article["text"]):
            sent = sent.strip()
            if len(sent) >= min_len:
                sentences.append({"text": sent, "source": article["title"]})
    return sentences
