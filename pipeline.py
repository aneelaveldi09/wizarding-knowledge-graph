"""
End-to-end ML pipeline runner.

Steps:
  1. Load Wikipedia corpus
  2. Run BERT NER on each sentence
  3. Run REBEL relation extraction
  4. Normalize entities + build graph
  5. Save graph to cache

Run: python pipeline.py [--no-rebel]  (--no-rebel uses spaCy SVO fallback)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_corpus, corpus_to_sentences
from src.ner_pipeline import batch_extract
from src.relation_extractor import extract_from_corpus
from src.graph_builder import build_from_triplets, save_graph, graph_stats

NER_CACHE = os.path.join("data", "ner_cache.json")
RE_CACHE = os.path.join("data", "re_cache.json")


def run(use_rebel: bool = True, use_cache: bool = True):
    t0 = time.time()

    # ── Step 1: Load corpus ──────────────────────────────────────────────────
    print("\n[1/4] Loading Wikipedia corpus…")
    corpus = load_corpus(use_cache=use_cache)
    sentences = corpus_to_sentences(corpus)
    print(f"  {len(corpus)} articles → {len(sentences)} sentences")

    # ── Step 2: BERT NER ─────────────────────────────────────────────────────
    print("\n[2/4] Running BERT NER (dslim/bert-base-NER)…")
    if use_cache and os.path.exists(NER_CACHE):
        with open(NER_CACHE) as f:
            enriched = json.load(f)
        print(f"  Loaded NER cache ({len(enriched)} sentences)")
    else:
        enriched = batch_extract(sentences, batch_size=16)
        with open(NER_CACHE, "w") as f:
            json.dump(enriched, f, indent=2)
        print(f"  NER complete — cached to {NER_CACHE}")

    total_ents = sum(len(s["entities"]) for s in enriched)
    print(f"  Extracted {total_ents} entity mentions")

    # ── Step 3: Relation Extraction ──────────────────────────────────────────
    method = "REBEL" if use_rebel else "spaCy SVO"
    print(f"\n[3/4] Running relation extraction ({method})…")
    if use_cache and os.path.exists(RE_CACHE):
        with open(RE_CACHE) as f:
            triplets = json.load(f)
        print(f"  Loaded RE cache ({len(triplets)} triplets)")
    else:
        triplets = extract_from_corpus(sentences, use_rebel=use_rebel)
        with open(RE_CACHE, "w") as f:
            json.dump(triplets, f, indent=2)
        print(f"  Extracted {len(triplets)} triplets — cached to {RE_CACHE}")

    # ── Step 4: Build + save graph ───────────────────────────────────────────
    print("\n[4/4] Building knowledge graph…")
    G = build_from_triplets(triplets)
    save_graph(G)
    stats = graph_stats(G)
    print(f"  Graph: {stats}")

    print(f"\nDone in {time.time() - t0:.1f}s")
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rebel", action="store_true", help="Use spaCy SVO instead of REBEL")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached data")
    args = parser.parse_args()
    run(use_rebel=not args.no_rebel, use_cache=not args.no_cache)
