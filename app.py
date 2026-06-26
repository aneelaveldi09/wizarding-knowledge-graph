"""Harry Potter Knowledge Graph — Streamlit Dashboard."""

import os
import sys
import tempfile

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.graph_builder import build_graph, get_subgraph, graph_stats
from src.extractor import extract_entities, SAMPLE_TEXT
from src.visualizer import build_pyvis, render_to_file
from data.hp_knowledge import ENTITY_COLORS

st.set_page_config(
    page_title="HP Knowledge Graph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body { background-color: #0E0E1A; }
.stMetric { background: #1A1A2E; border-radius: 8px; padding: 10px; }
h1 { color: #FFD700; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Harry_Potter_wordmark.svg/320px-Harry_Potter_wordmark.svg.png",
    use_column_width=True,
)
st.sidebar.title("Graph Controls")

entity_filter = st.sidebar.multiselect(
    "Show entity types",
    ["Character", "Spell", "Location", "Event", "Object"],
    default=["Character", "Spell", "Location", "Event", "Object"],
)

focus_node = st.sidebar.text_input(
    "Focus on entity (leave blank for full graph)",
    value="",
    placeholder="e.g. harry_potter",
)
focus_depth = st.sidebar.slider("Focus depth (hops)", 1, 3, 1)

physics_on = st.sidebar.checkbox("Enable physics simulation", value=True)

relation_filter = st.sidebar.text_input(
    "Filter by relationship keyword",
    value="",
    placeholder="e.g. fought, teaches",
)

# ── Build graph ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_graph():
    return build_graph()

G_full = get_graph()

# Apply entity type filter
kept_nodes = [n for n, d in G_full.nodes(data=True) if d.get("entity_type") in entity_filter]
G = G_full.subgraph(kept_nodes).copy()

# Apply relation keyword filter
if relation_filter.strip():
    kw = relation_filter.strip().lower()
    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if kw not in d.get("relation", "").lower()
    ]
    G.remove_edges_from(edges_to_remove)

# Apply focus
if focus_node.strip() and focus_node.strip() in G.nodes:
    G = get_subgraph(G, focus_node.strip(), depth=focus_depth)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("⚡ Harry Potter Knowledge Graph")
st.markdown("An interactive graph of characters, spells, locations, events, and objects from the wizarding world.")

# ── Metrics ───────────────────────────────────────────────────────────────────

stats = graph_stats(G)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Nodes", stats["nodes"])
c2.metric("Total Edges", stats["edges"])
c3.metric("Characters", stats["characters"])
c4.metric("Spells", stats["spells"])
c5.metric("Locations", stats["locations"])
c6.metric("Events + Objects", stats["events"] + stats["objects"])

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["🕸️ Knowledge Graph", "📊 Analytics", "🔍 NLP Extractor", "📋 Entity Table"])

# ── Tab 1: Graph ──────────────────────────────────────────────────────────────

with tab1:
    if G.number_of_nodes() == 0:
        st.warning("No nodes match the current filters.")
    else:
        net = build_pyvis(G, height="700px", physics=physics_on)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            tmp_path = f.name
        net.save_graph(tmp_path)
        with open(tmp_path, "r") as f:
            html = f.read()
        os.unlink(tmp_path)
        components.html(html, height=720, scrolling=False)

    st.caption(
        "**Legend** — 🔴 Character · 💠 Spell · 🟦 Location · ⭐ Event · 🔺 Object  |  "
        "Node size = number of connections  |  Click a node to highlight its neighbours."
    )

# ── Tab 2: Analytics ──────────────────────────────────────────────────────────

with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Entity Distribution")
        type_counts = {}
        for _, d in G.nodes(data=True):
            t = d.get("entity_type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        fig_pie = px.pie(
            names=list(type_counts.keys()),
            values=list(type_counts.values()),
            color=list(type_counts.keys()),
            color_discrete_map=ENTITY_COLORS,
            title="Entity Type Breakdown",
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Most Connected Entities (Top 15)")
        degree_data = [
            {"Entity": G.nodes[n].get("label", n), "Type": G.nodes[n].get("entity_type", "?"), "Connections": G.degree(n)}
            for n in G.nodes
        ]
        df_degree = pd.DataFrame(degree_data).sort_values("Connections", ascending=False).head(15)
        fig_bar = px.bar(
            df_degree, x="Connections", y="Entity", orientation="h",
            color="Type", color_discrete_map=ENTITY_COLORS,
            title="Top 15 Nodes by Degree",
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Relationship Type Frequency")
    rel_counts: dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        rel = d.get("relation", "related_to")
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    df_rel = pd.DataFrame(
        [{"Relationship": k, "Count": v} for k, v in rel_counts.items()]
    ).sort_values("Count", ascending=False)
    fig_rel = px.bar(df_rel, x="Relationship", y="Count", title="Edge Relationship Frequencies", color="Count",
                     color_continuous_scale="Reds")
    st.plotly_chart(fig_rel, use_container_width=True)

# ── Tab 3: NLP Extractor ──────────────────────────────────────────────────────

with tab3:
    st.subheader("NLP Entity Extraction")
    st.markdown(
        "Paste any Harry Potter text below and the extractor will identify entities using keyword matching "
        "and (if spaCy is installed) a trained NER model."
    )
    user_text = st.text_area("Input text", value=SAMPLE_TEXT, height=200)
    if st.button("Extract Entities"):
        with st.spinner("Extracting…"):
            entities = extract_entities(user_text)
        if entities:
            df_ents = pd.DataFrame(entities)[["label", "type", "mentions"]]
            df_ents.columns = ["Entity", "Type", "Mentions"]

            col_x, col_y = st.columns([2, 1])
            with col_x:
                fig_ent = px.bar(
                    df_ents.head(20), x="Mentions", y="Entity", orientation="h",
                    color="Type", color_discrete_map=ENTITY_COLORS,
                    title="Extracted Entities by Mention Count",
                )
                fig_ent.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_ent, use_container_width=True)
            with col_y:
                st.dataframe(df_ents, use_container_width=True)
        else:
            st.info("No known HP entities found in the text.")

# ── Tab 4: Entity Table ───────────────────────────────────────────────────────

with tab4:
    st.subheader("All Entities")
    rows = []
    for n, d in G.nodes(data=True):
        row = {"ID": n, "Label": d.get("label", n), "Type": d.get("entity_type", "?")}
        row["In-degree"] = G.in_degree(n)
        row["Out-degree"] = G.out_degree(n)
        rows.append(row)
    df_nodes = pd.DataFrame(rows).sort_values(["Type", "Label"])
    st.dataframe(df_nodes, use_container_width=True, hide_index=True)

    st.subheader("All Relationships")
    edge_rows = []
    for u, v, d in G.edges(data=True):
        edge_rows.append({
            "From": G.nodes[u].get("label", u),
            "Relationship": d.get("relation", "?"),
            "To": G.nodes[v].get("label", v),
        })
    df_edges = pd.DataFrame(edge_rows).sort_values("Relationship")
    st.dataframe(df_edges, use_container_width=True, hide_index=True)
