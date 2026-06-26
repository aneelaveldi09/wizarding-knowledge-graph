"""
Named Entity Recognition using dslim/bert-base-NER (HuggingFace transformer).
Model: fine-tuned BERT on CoNLL-2003 — extracts PER, LOC, ORG, MISC entities.
"""

from __future__ import annotations
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "dslim/bert-base-NER"

_ner_pipeline = None


def _get_device() -> int | str:
    if torch.cuda.is_available():
        return 0
    # MPS (Apple Silicon) — transformers pipeline accepts device="mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return -1


def load_ner_model():
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline
    print(f"Loading NER model: {MODEL_NAME}")
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    _ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="max",
        device=device,
    )
    print(f"  NER model loaded on device: {device}")
    return _ner_pipeline


def extract_entities(text: str) -> list[dict]:
    """
    Run BERT NER on text.
    Returns list of: {word, entity_group, score, start, end}
    entity_group: PER | LOC | ORG | MISC
    """
    ner = load_ner_model()
    results = ner(text)
    return [
        {
            "word": r["word"].strip(),
            "entity_group": r["entity_group"],
            "score": round(float(r["score"]), 4),
            "start": r["start"],
            "end": r["end"],
        }
        for r in results
        if r["score"] > 0.7 and len(r["word"].strip()) > 1
    ]


def batch_extract(sentences: list[dict], batch_size: int = 16) -> list[dict]:
    """
    Run NER over a list of sentence dicts (with 'text' and 'source' keys).
    Returns sentence dicts enriched with 'entities' key.
    """
    ner = load_ner_model()
    texts = [s["text"] for s in sentences]
    enriched = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_results = ner(batch)
        for j, results in enumerate(batch_results):
            sent = dict(sentences[i + j])
            sent["entities"] = [
                {
                    "word": r["word"].strip(),
                    "entity_group": r["entity_group"],
                    "score": round(float(r["score"]), 4),
                }
                for r in results
                if r["score"] > 0.7 and len(r["word"].strip()) > 1
            ]
            enriched.append(sent)
    return enriched
