<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/c/c4/Wizarding_World_of_Harry_Potter_Castle.jpg" width="100%" style="border-radius:12px" alt="Wizarding World of Harry Potter"/>

# Wizarding Knowledge Graph

**What if you could map the entire wizarding world using machine learning?**

I built this because I've been obsessed with Harry Potter since I was a kid, and now that I'm deep into ML, I wanted to combine both. This project pulls real text from Wikipedia about HP characters, spells, locations, and events — then runs it through a BERT NER model and a REBEL seq2seq transformer to extract every relationship and render it as a live, interactive knowledge graph.

No hardcoded data. Everything you see in the graph was extracted by the models.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph-0769AD?style=flat)

</div>

## What this actually does

Most "knowledge graph" projects I found online just hardcode a JSON file of relationships and call it a graph. That's not ML — that's a dictionary. I wanted to build one where the machine figures out the relationships on its own.

The pipeline works like this:

```
Wikipedia / HP Fandom Wiki
        ↓
  50 articles · 2,179 sentences of real HP text
        ↓
  dslim/bert-base-NER  ← BERT fine-tuned on CoNLL-2003
  Extracts: Harry Potter (PER), Hogwarts (ORG), Azkaban (LOC)...
        ↓
  Babelscape/rebel-large  ← BART seq2seq transformer
  Extracts: (Harry Potter) → attended → (Hogwarts)
            (Sirius Black) → imprisoned in → (Azkaban)
            (Hermione Granger) → used → (Time-Turner)
        ↓
  NetworkX Knowledge Graph
        ↓
  Interactive Pyvis dashboard
```

## Tech stack

| Component | What I used | Why |
|---|---|---|
| NER | `dslim/bert-base-NER` | BERT fine-tuned on CoNLL-2003, strong on people + places |
| Relation extraction | `Babelscape/rebel-large` | End-to-end seq2seq: text in, triplets out |
| Graph | NetworkX + Pyvis | Lightweight, interactive, runs in browser |
| Dashboard | Streamlit | Fast to iterate, good for ML demos |
| Acceleration | Apple MPS / CUDA | Auto-detected, falls back to CPU |

## Running it yourself

```bash
git clone https://github.com/aneelaveldi09/wizarding-knowledge-graph.git
cd wizarding-knowledge-graph
pip install -r requirements.txt

# Option 1: run the full ML pipeline (downloads REBEL ~1.5 GB on first run)
python pipeline.py

# Option 2: launch the dashboard (uses pre-built cached graph instantly)
streamlit run app.py
```

The pre-built caches (Wikipedia text, NER results, graph) are committed to the repo so the app loads immediately without running the models. The Streamlit dashboard also lets you run REBEL live on any text you paste in.

## Dashboard tabs

**Knowledge Graph** — interactive Pyvis graph. Click any node, filter by entity type, focus on a character to see their connections.

**BERT NER** — paste any Harry Potter text, watch the model extract characters and locations in real time with confidence scores.

**REBEL Triplets** — test the relation extractor on custom sentences. See exactly what (head, relation, tail) triplets the model pulls out.

**Analytics** — PageRank centrality, degree distribution, entity type breakdown, top relation frequencies.

## What the models actually find

Running REBEL on real Wikipedia text, some of the triplets it extracts:

- Harry Potter → student → Hogwarts School of Witchcraft and Wizardry
- Albus Dumbledore → headmaster → Hogwarts
- Sirius Black → imprisoned → Azkaban
- Hermione Granger → married → Ron Weasley
- Voldemort → killed → Cedric Diggory
- Severus Snape → member → Order of the Phoenix

These come purely from model inference on raw text, not from any lookup table.

<div align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/Diagon_Alley%2C_The_making_of_Harry_Potter_%28Ank_Kumar%2C_Infosys%29_02.jpg" width="680" alt="Diagon Alley — Warner Bros Studio Tour"/>
<br>
<sub>Diagon Alley — Warner Bros. Studio Tour, The Making of Harry Potter</sub>
</div>

## Project structure

```
wizarding-knowledge-graph/
├── app.py                    # Streamlit dashboard
├── pipeline.py               # CLI pipeline runner
├── src/
│   ├── data_loader.py        # Wikipedia + HP Fandom Wiki fetcher
│   ├── ner_pipeline.py       # BERT NER inference
│   ├── relation_extractor.py # REBEL + spaCy SVO fallback
│   ├── entity_normalizer.py  # surface form → canonical ID
│   ├── graph_builder.py      # NetworkX graph construction
│   └── visualizer.py         # Pyvis rendering
└── data/
    ├── entity_aliases.py     # HP entity alias map (used for normalization only)
    ├── wiki_cache.json       # cached Wikipedia text
    ├── ner_cache.json        # cached BERT NER output
    ├── re_cache.json         # cached REBEL triplets
    └── graph_cache.json      # pre-built knowledge graph
```

---

**Built by [Aneela Veldi](https://github.com/aneelaveldi09)**
