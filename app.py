"""
Harry Potter Knowledge Graph — Streamlit Dashboard
ML Pipeline: Fandom Wiki text → BERT NER → REBEL triplets → Knowledge Graph
"""

import json
import os
import sys
import tempfile

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_corpus, corpus_to_sentences
from src.ner_pipeline import batch_extract, extract_entities
from src.relation_extractor import extract_triplets_rebel, extract_from_corpus
from src.graph_builder import (
    build_from_triplets, save_graph, load_graph, graph_stats, get_subgraph,
    GRAPH_CACHE,
)
from src.entity_normalizer import get_entity_color
from src.visualizer import build_pyvis
from data.entity_aliases import ENTITY_COLORS

NER_CACHE  = os.path.join("data", "ner_cache.json")
RE_CACHE   = os.path.join("data", "re_cache.json")

st.set_page_config(
    page_title="HP Knowledge Graph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stMetric { background: #1A1A2E; border-radius: 8px; padding: 10px; }
.pipeline-step { background: #1e1e2e; border-left: 4px solid #FF6B6B;
                 padding: 10px 16px; border-radius: 4px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("⚡ HP Knowledge Graph")
st.sidebar.markdown("**ML Pipeline**")
st.sidebar.markdown("""
1. 🌐 HP Fandom Wiki text
2. 🤖 BERT NER (`dslim/bert-base-NER`)
3. 🔗 REBEL (`Babelscape/rebel-large`)
4. 🕸️ NetworkX graph
""")
st.sidebar.divider()

entity_filter = st.sidebar.multiselect(
    "Entity types to show",
    ["Character", "Location", "Spell", "Object", "Event", "Organization", "House"],
    default=["Character", "Location", "Spell", "Object", "Event", "Organization", "House"],
)
focus_node = st.sidebar.text_input("Focus on node ID", placeholder="e.g. harry_potter")
focus_depth = st.sidebar.slider("Focus depth", 1, 3, 1)
physics_on = st.sidebar.checkbox("Physics simulation", value=True)

# ── Load / run pipeline ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_corpus():
    return load_corpus(use_cache=True)

@st.cache_resource(show_spinner=False)
def get_sentences(corpus):
    return corpus_to_sentences(corpus)

@st.cache_resource(show_spinner=False)
def get_ner_results(sentences_json: str):
    sentences = json.loads(sentences_json)
    if os.path.exists(NER_CACHE):
        with open(NER_CACHE) as f:
            return json.load(f)
    enriched = batch_extract(sentences, batch_size=16)
    with open(NER_CACHE, "w") as f:
        json.dump(enriched, f)
    return enriched

@st.cache_resource(show_spinner=False)
def get_triplets(sentences_json: str, use_rebel: bool):
    sentences = json.loads(sentences_json)
    if os.path.exists(RE_CACHE):
        with open(RE_CACHE) as f:
            return json.load(f)
    triplets = extract_from_corpus(sentences, use_rebel=use_rebel)
    with open(RE_CACHE, "w") as f:
        json.dump(triplets, f)
    return triplets

@st.cache_resource(show_spinner=False)
def get_graph(_triplets_json: str):
    G = load_graph()
    if G is not None and G.number_of_nodes() > 0:
        return G
    triplets = json.loads(_triplets_json)
    G = build_from_triplets(triplets)
    save_graph(G)
    return G

# ── Header ────────────────────────────────────────────────────────────────────

st.title("⚡ Harry Potter Knowledge Graph")
st.markdown(
    "Built **entirely from real text** — HP Fandom Wiki articles processed through "
    "a BERT NER model + REBEL seq2seq transformer. No hardcoded data."
)

# ── Pipeline runner UI ────────────────────────────────────────────────────────

with st.expander("▶  Run / refresh ML pipeline", expanded=not os.path.exists(GRAPH_CACHE)):
    use_rebel = st.checkbox("Use REBEL for relation extraction (recommended, ~1.5 GB first download)", value=True)
    force_refresh = st.checkbox("Force re-run (ignore cache)")
    run_btn = st.button("🚀 Run Pipeline", type="primary")

    if run_btn or not os.path.exists(GRAPH_CACHE):
        if force_refresh:
            for p in [NER_CACHE, RE_CACHE, GRAPH_CACHE,
                      os.path.join("data", "wiki_cache.json")]:
                if os.path.exists(p): os.remove(p)
            st.cache_resource.clear()

        prog = st.progress(0, "Step 1/4 — Fetching HP Fandom Wiki articles…")
        corpus = load_corpus(use_cache=not force_refresh)
        sentences = corpus_to_sentences(corpus)
        prog.progress(25, f"Step 2/4 — BERT NER on {len(sentences)} sentences…")

        enriched = get_ner_results(json.dumps(sentences))
        total_ents = sum(len(s["entities"]) for s in enriched)
        prog.progress(50, f"Step 3/4 — {'REBEL' if use_rebel else 'spaCy SVO'} relation extraction…")

        triplets = get_triplets(json.dumps(sentences), use_rebel=use_rebel)
        prog.progress(75, f"Step 4/4 — Building graph from {len(triplets)} triplets…")

        G_built = build_from_triplets(triplets)
        save_graph(G_built)
        st.cache_resource.clear()
        prog.progress(100, "Done!")
        st.success(f"Pipeline complete — {G_built.number_of_nodes()} nodes, "
                   f"{G_built.number_of_edges()} edges extracted by ML models.")

# ── Load cached graph ─────────────────────────────────────────────────────────

G_full = load_graph()
if G_full is None or G_full.number_of_nodes() == 0:
    st.info("No graph found. Expand 'Run / refresh ML pipeline' above and click Run Pipeline.")
    st.stop()

# Filter by entity type
kept = [n for n, d in G_full.nodes(data=True) if d.get("entity_type") in entity_filter]
G = G_full.subgraph(kept).copy()

# Focus
if focus_node.strip() and focus_node.strip() in G.nodes:
    G = get_subgraph(G, focus_node.strip(), depth=focus_depth)

# ── Metrics ───────────────────────────────────────────────────────────────────

stats = graph_stats(G)
cols = st.columns(len(stats))
for col, (k, v) in zip(cols, stats.items()):
    col.metric(k.replace("_", " ").title(), v)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_graph, tab_ner, tab_re, tab_analytics, tab_table = st.tabs(
    ["🕸️ Knowledge Graph", "🤖 BERT NER", "🔗 REBEL Triplets", "📊 Analytics", "📋 Tables"]
)

# ── Tab: Graph ────────────────────────────────────────────────────────────────

with tab_graph:
    if G.number_of_nodes() == 0:
        st.warning("No nodes match the selected filters.")
    else:
        net = build_pyvis(G, physics=physics_on)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            tmp = f.name
        net.save_graph(tmp)
        with open(tmp) as f:
            html = f.read()
        os.unlink(tmp)
        components.html(html, height=700, scrolling=False)
    st.caption(
        "**Shapes** — ● Character  ■ Location  ◆ Spell  ▲ Object  ★ Event  ⬡ Organization  "
        "| Node size = degree | All relationships extracted by REBEL from real HP Wiki text."
    )

# ── Tab: NER ─────────────────────────────────────────────────────────────────

with tab_ner:
    st.subheader("BERT Named Entity Recognition")
    st.markdown(
        "Model: [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER) — "
        "BERT fine-tuned on CoNLL-2003. Extracts **PER** (person), **LOC** (location), "
        "**ORG** (organization), **MISC** from raw HP Fandom Wiki text."
    )

    col_input, col_result = st.columns([1, 1])
    with col_input:
        demo_text = st.text_area(
            "Try it on any text",
            value=(
                "Harry Potter and Hermione Granger attended Hogwarts School of Witchcraft "
                "and Wizardry. Albus Dumbledore was the headmaster until Severus Snape "
                "betrayed him. Lord Voldemort was defeated at the Battle of Hogwarts."
            ),
            height=160,
        )
        run_ner = st.button("Extract entities", key="ner_btn")

    if run_ner:
        with st.spinner("Running BERT NER…"):
            ents = extract_entities(demo_text)
        with col_result:
            if ents:
                df_e = pd.DataFrame(ents)[["word", "entity_group", "score"]]
                df_e.columns = ["Entity", "Type", "Confidence"]
                st.dataframe(df_e, use_container_width=True, hide_index=True)
                fig = px.bar(
                    df_e, x="Confidence", y="Entity", orientation="h",
                    color="Type", title="Extracted Entities by Confidence",
                    color_discrete_map={"PER": "#FF6B6B", "LOC": "#45B7D1",
                                        "ORG": "#C39BD3", "MISC": "#98D8C8"},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No entities found.")

    # Show cached NER stats if available
    if os.path.exists(NER_CACHE):
        st.divider()
        st.markdown("**Corpus-wide NER results** (from pipeline run)")
        with open(NER_CACHE) as f:
            ner_data = json.load(f)
        all_ents = [e for s in ner_data for e in s["entities"]]
        if all_ents:
            df_all = pd.DataFrame(all_ents)
            type_counts = df_all["entity_group"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            col_a, col_b = st.columns(2)
            with col_a:
                fig_t = px.pie(type_counts, names="Type", values="Count",
                               title="Entity Type Distribution", hole=0.4,
                               color_discrete_map={"PER": "#FF6B6B", "LOC": "#45B7D1",
                                                   "ORG": "#C39BD3", "MISC": "#98D8C8"})
                st.plotly_chart(fig_t, use_container_width=True)
            with col_b:
                top_ents = (df_all.groupby("word")["score"]
                            .agg(["count", "mean"])
                            .sort_values("count", ascending=False)
                            .head(20)
                            .reset_index())
                top_ents.columns = ["Entity", "Mentions", "Avg Confidence"]
                fig_top = px.bar(top_ents, x="Mentions", y="Entity", orientation="h",
                                 title="Top 20 Most-Mentioned Entities")
                fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_top, use_container_width=True)

# ── Tab: REBEL Triplets ───────────────────────────────────────────────────────

with tab_re:
    st.subheader("REBEL Relation Extraction")
    st.markdown(
        "Model: [`Babelscape/rebel-large`](https://huggingface.co/Babelscape/rebel-large) — "
        "BART seq2seq transformer. Given raw text, it generates structured **(head, relation, tail)** "
        "triplets that directly populate the knowledge graph."
    )

    demo_re_text = st.text_area(
        "Test REBEL on any sentence",
        value=(
            "Harry Potter attended Hogwarts where Albus Dumbledore was the headmaster. "
            "Hermione Granger used a Time-Turner to travel back in time. "
            "Severus Snape taught Potions and was a member of the Order of the Phoenix."
        ),
        height=120,
        key="re_text",
    )
    run_re = st.button("Extract triplets", key="re_btn")

    if run_re:
        with st.spinner("Running REBEL (this downloads ~1.5 GB on first run)…"):
            try:
                triplets = extract_triplets_rebel(demo_re_text)
                if triplets:
                    df_t = pd.DataFrame(triplets)
                    st.success(f"Extracted {len(triplets)} triplet(s)")
                    st.dataframe(df_t[["head", "relation", "tail"]], use_container_width=True, hide_index=True)
                else:
                    st.warning("No triplets extracted from this text.")
            except Exception as e:
                st.error(f"REBEL error: {e}")

    if os.path.exists(RE_CACHE):
        st.divider()
        st.markdown("**Triplets extracted from corpus**")
        with open(RE_CACHE) as f:
            all_triplets = json.load(f)
        df_re = pd.DataFrame(all_triplets)
        if not df_re.empty and "relation" in df_re.columns:
            rel_counts = df_re["relation"].value_counts().head(25).reset_index()
            rel_counts.columns = ["Relation", "Count"]
            fig_rel = px.bar(rel_counts, x="Count", y="Relation", orientation="h",
                             title="Top 25 Relation Types Extracted by REBEL",
                             color="Count", color_continuous_scale="Reds")
            fig_rel.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_rel, use_container_width=True)
            st.dataframe(df_re[["head", "relation", "tail", "source"]].head(100),
                         use_container_width=True, hide_index=True)

# ── Tab: Analytics ────────────────────────────────────────────────────────────

with tab_analytics:
    st.subheader("Graph Analytics")
    col1, col2 = st.columns(2)

    with col1:
        type_counts = {}
        for _, d in G.nodes(data=True):
            t = d.get("entity_type", "Other")
            type_counts[t] = type_counts.get(t, 0) + 1
        fig_pie = px.pie(
            names=list(type_counts.keys()),
            values=list(type_counts.values()),
            color=list(type_counts.keys()),
            color_discrete_map=ENTITY_COLORS,
            title="Entity Type Distribution",
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        degree_data = [
            {"Entity": G.nodes[n].get("label", n),
             "Type": G.nodes[n].get("entity_type", "?"),
             "Degree": G.degree(n)}
            for n in G.nodes
        ]
        df_deg = pd.DataFrame(degree_data).sort_values("Degree", ascending=False).head(20)
        fig_deg = px.bar(
            df_deg, x="Degree", y="Entity", orientation="h",
            color="Type", color_discrete_map=ENTITY_COLORS,
            title="Top 20 Most-Connected Entities",
        )
        fig_deg.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_deg, use_container_width=True)

    # Centrality
    st.markdown("**PageRank Centrality** (most important nodes by link structure)")
    pr = nx.pagerank(G, alpha=0.85)
    df_pr = pd.DataFrame(
        [{"Entity": G.nodes[n].get("label", n), "Type": G.nodes[n].get("entity_type", "?"),
          "PageRank": round(v, 5)} for n, v in pr.items()]
    ).sort_values("PageRank", ascending=False).head(15)
    fig_pr = px.bar(df_pr, x="PageRank", y="Entity", orientation="h",
                    color="Type", color_discrete_map=ENTITY_COLORS,
                    title="Top 15 Nodes by PageRank")
    fig_pr.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_pr, use_container_width=True)

# ── Tab: Tables ───────────────────────────────────────────────────────────────

with tab_table:
    st.subheader("Nodes")
    node_rows = [
        {"ID": n, "Label": d.get("label", n), "Type": d.get("entity_type", "?"),
         "In": G.in_degree(n), "Out": G.out_degree(n), "Degree": G.degree(n)}
        for n, d in G.nodes(data=True)
    ]
    st.dataframe(pd.DataFrame(node_rows).sort_values("Degree", ascending=False),
                 use_container_width=True, hide_index=True)

    st.subheader("Edges (ML-extracted relationships)")
    edge_rows = [
        {"From": G.nodes[u].get("label", u),
         "Relation": d.get("relation", "?").replace("_", " "),
         "To": G.nodes[v].get("label", v),
         "Weight": d.get("weight", 1),
         "Source": ", ".join(d.get("sources", [])[:2])}
        for u, v, d in G.edges(data=True)
    ]
    st.dataframe(pd.DataFrame(edge_rows).sort_values("Weight", ascending=False),
                 use_container_width=True, hide_index=True)
