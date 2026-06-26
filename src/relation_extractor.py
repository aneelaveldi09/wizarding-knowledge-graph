"""
Relation extraction using two complementary ML approaches:

1. REBEL (Babelscape/rebel-large) — BART-based seq2seq transformer.
   Reads a sentence and generates structured (head, relation, tail) triplets.
   This is the primary extractor.

2. spaCy SVO (Subject-Verb-Object) — neural dependency parser.
   Used as lightweight fallback / complement.
"""

from __future__ import annotations
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REBEL_MODEL = "Babelscape/rebel-large"

_rebel_tokenizer = None
_rebel_model = None


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rebel():
    global _rebel_tokenizer, _rebel_model
    if _rebel_model is not None:
        return _rebel_tokenizer, _rebel_model
    print(f"Loading REBEL model: {REBEL_MODEL}  (~1.5 GB, first run downloads)")
    device = _get_device()
    _rebel_tokenizer = AutoTokenizer.from_pretrained(REBEL_MODEL)
    _rebel_model = AutoModelForSeq2SeqLM.from_pretrained(REBEL_MODEL).to(device)
    _rebel_model.eval()
    print(f"  REBEL loaded on device: {device}")
    return _rebel_tokenizer, _rebel_model


def _parse_rebel_output(text: str) -> list[dict]:
    """
    REBEL output format:
      <triplet> SUBJECT <subj> OBJECT <obj> RELATION <triplet> ...
    """
    triplets = []
    subject, obj, relation = "", "", ""
    state = "x"

    tokens = (
        text.replace("<s>", "")
            .replace("<pad>", "")
            .replace("</s>", "")
            .split()
    )
    for token in tokens:
        if token == "<triplet>":
            if relation:
                triplets.append({"head": subject.strip(), "relation": relation.strip(), "tail": obj.strip()})
                relation = ""
            subject = ""
            state = "subject"
        elif token == "<subj>":
            if relation:
                triplets.append({"head": subject.strip(), "relation": relation.strip(), "tail": obj.strip()})
            obj = ""
            state = "object"
        elif token == "<obj>":
            relation = ""
            state = "relation"
        else:
            if state == "subject":
                subject += " " + token
            elif state == "object":
                obj += " " + token
            elif state == "relation":
                relation += " " + token

    if subject and relation and obj:
        triplets.append({"head": subject.strip(), "relation": relation.strip(), "tail": obj.strip()})

    return [t for t in triplets if t["head"] and t["relation"] and t["tail"]]


def extract_triplets_rebel(text: str, max_input_length: int = 512) -> list[dict]:
    """Run REBEL on a text chunk and return (head, relation, tail) triplets."""
    tokenizer, model = load_rebel()
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        max_length=max_input_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=3,
            max_length=512,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return _parse_rebel_output(decoded)


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """Split text into sentence-aligned chunks for REBEL inference."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks


def extract_from_corpus(sentences: list[dict], use_rebel: bool = True) -> list[dict]:
    """
    Run relation extraction over a list of sentence dicts.
    Each sentence dict: {'text': str, 'source': str}
    Returns list of triplet dicts with 'source' metadata.
    """
    all_triplets = []

    if use_rebel:
        # Group sentences into ~400-char chunks for REBEL
        buffer, source = "", ""
        for sent in sentences:
            if len(buffer) + len(sent["text"]) < 400:
                buffer += " " + sent["text"]
                source = sent["source"]
            else:
                if buffer:
                    for t in extract_triplets_rebel(buffer):
                        t["source"] = source
                        all_triplets.append(t)
                buffer = sent["text"]
                source = sent["source"]
        if buffer:
            for t in extract_triplets_rebel(buffer):
                t["source"] = source
                all_triplets.append(t)
    else:
        # spaCy SVO fallback
        all_triplets.extend(_svo_fallback(sentences))

    return all_triplets


def _svo_fallback(sentences: list[dict]) -> list[dict]:
    """spaCy neural dependency parser → SVO triplets (fallback)."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return []

    triplets = []
    for sent in sentences:
        doc = nlp(sent["text"])
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                objects = [c for c in token.children if c.dep_ in ("dobj", "attr", "pobj", "prep")]
                for subj in subjects:
                    for obj in objects:
                        triplets.append({
                            "head": subj.text,
                            "relation": token.lemma_,
                            "tail": obj.text,
                            "source": sent["source"],
                        })
    return triplets
