"""
NLP entity extraction using spaCy.
Extracts characters, locations, and other named entities from raw HP text.
"""

from __future__ import annotations
import re
from typing import Optional

_nlp = None

KNOWN_ENTITIES = {
    "Harry Potter": "Character",
    "Hermione Granger": "Character",
    "Ron Weasley": "Character",
    "Albus Dumbledore": "Character",
    "Severus Snape": "Character",
    "Lord Voldemort": "Character",
    "Voldemort": "Character",
    "Draco Malfoy": "Character",
    "Neville Longbottom": "Character",
    "Luna Lovegood": "Character",
    "Sirius Black": "Character",
    "Remus Lupin": "Character",
    "Bellatrix Lestrange": "Character",
    "Minerva McGonagall": "Character",
    "Rubeus Hagrid": "Character",
    "Hogwarts": "Location",
    "Diagon Alley": "Location",
    "Hogsmeade": "Location",
    "Azkaban": "Location",
    "Ministry of Magic": "Location",
    "Forbidden Forest": "Location",
    "Gringotts": "Location",
    "Expelliarmus": "Spell",
    "Avada Kedavra": "Spell",
    "Expecto Patronum": "Spell",
    "Wingardium Leviosa": "Spell",
    "Lumos": "Spell",
    "Stupefy": "Spell",
    "Crucio": "Spell",
    "Time Turner": "Object",
    "Elder Wand": "Object",
    "Invisibility Cloak": "Object",
    "Philosopher's Stone": "Object",
    "Marauder's Map": "Object",
}


def _load_spacy() -> Optional[object]:
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except Exception:
        return None


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities from text.
    Falls back to regex-based matching if spaCy is unavailable.
    """
    found: dict[str, dict] = {}

    # Keyword matching against known HP entities
    for name, etype in KNOWN_ENTITIES.items():
        pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            key = name.lower().replace(" ", "_")
            if key not in found:
                found[key] = {
                    "id": key,
                    "label": name,
                    "type": etype,
                    "mentions": 0,
                    "positions": [],
                }
            found[key]["mentions"] += 1
            found[key]["positions"].append(match.start())

    # spaCy NER for additional entities
    nlp = _load_spacy()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "GPE", "LOC", "ORG", "FAC"):
                key = ent.text.lower().replace(" ", "_")
                if key not in found:
                    etype_map = {
                        "PERSON": "Character",
                        "GPE": "Location",
                        "LOC": "Location",
                        "ORG": "Organization",
                        "FAC": "Location",
                    }
                    found[key] = {
                        "id": key,
                        "label": ent.text,
                        "type": etype_map.get(ent.label_, "Unknown"),
                        "mentions": 0,
                        "positions": [],
                    }
                found[key]["mentions"] += 1
                found[key]["positions"].append(ent.start_char)

    return sorted(found.values(), key=lambda x: -x["mentions"])


SAMPLE_TEXT = """
Harry Potter and his best friends Hermione Granger and Ron Weasley returned to Hogwarts
for their third year. Professor Remus Lupin taught Defense Against the Dark Arts, while
Severus Snape continued to teach Potions with his usual disdain for Harry.

During the year, Sirius Black escaped from Azkaban, causing panic across the wizarding world.
Hermione used a Time Turner to attend multiple classes simultaneously, a fact that would later
prove crucial when she and Harry rescued both Sirius and the hippogriff Buckbeak.

Lord Voldemort, though not yet returned to full power, manipulated events through his loyal
servant Peter Pettigrew. At Hogwarts, in the Forbidden Forest and the Shrieking Shack,
the truth about Sirius's innocence was finally revealed. Hermione cast Expecto Patronum
to repel the Dementors, while Harry used Expelliarmus against Snape.
"""
