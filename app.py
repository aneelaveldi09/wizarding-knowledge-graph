"""Harry Potter Knowledge Graph — Streamlit Dashboard"""

import json, os, sys, re
import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader    import load_corpus, corpus_to_sentences
from src.graph_builder  import build_from_triplets, save_graph, load_graph, graph_stats, get_subgraph, GRAPH_CACHE
from src.visualizer     import cytoscape_html, ENTITY_COLORS

# ML models loaded lazily — app works from cache without them
try:
    from src.ner_pipeline       import batch_extract, extract_entities
    from src.relation_extractor import extract_triplets_rebel, extract_from_corpus
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

NER_CACHE = os.path.join("data", "ner_cache.json")
RE_CACHE  = os.path.join("data", "re_cache.json")

st.set_page_config(
    page_title="HP Knowledge Graph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Base ── */
.stApp { background: #05050f; color: #e8e8e8; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1e 0%, #05050f 100%);
    border-right: 1px solid rgba(201,162,39,0.2);
}
[data-testid="stSidebar"] * { color: #e8e8e8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #c9a227 !important; }

/* ── Typography ── */
h1 { font-family: 'Cinzel', serif !important; color: #c9a227 !important;
     text-shadow: 0 0 40px rgba(201,162,39,0.5); letter-spacing: 2px; }
h2 { font-family: 'Cinzel', serif !important; color: #c9a227 !important; }
h3 { color: #d4af37 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d0d2b 0%, #141438 100%);
    border: 1px solid rgba(201,162,39,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 0 20px rgba(201,162,39,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
}
[data-testid="stMetricLabel"] p { color: #c9a227 !important; font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 2rem !important; font-weight: 700 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,13,43,0.6);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(201,162,39,0.15);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #888 !important;
    border-radius: 7px;
    padding: 8px 20px;
    font-size: 0.9rem;
    border: none !important;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { color: #c9a227 !important; background: rgba(201,162,39,0.08); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(201,162,39,0.2), rgba(201,162,39,0.1)) !important;
    color: #c9a227 !important;
    border: 1px solid rgba(201,162,39,0.35) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #c9a227, #a07d1c) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(201,162,39,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(201,162,39,0.5) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input, .stTextArea textarea {
    background: #0d0d2b !important;
    border: 1px solid rgba(201,162,39,0.3) !important;
    color: #e8e8e8 !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: #c9a227 !important;
    box-shadow: 0 0 0 2px rgba(201,162,39,0.2) !important;
}

/* ── Multiselect ── */
.stMultiSelect [data-baseweb="select"] {
    background: #0d0d2b !important;
    border: 1px solid rgba(201,162,39,0.3) !important;
    border-radius: 8px !important;
}
.stMultiSelect [data-baseweb="tag"] { background: rgba(201,162,39,0.2) !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #c9a227 !important; }
.stSlider [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] { background: #c9a227 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(13,13,43,0.6) !important;
    border: 1px solid rgba(201,162,39,0.2) !important;
    border-radius: 8px !important;
    color: #c9a227 !important;
}
.streamlit-expanderContent {
    border: 1px solid rgba(201,162,39,0.15) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    background: rgba(5,5,15,0.6) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid rgba(201,162,39,0.2); border-radius: 8px; overflow: hidden; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #c9a227, #ffd700) !important; border-radius: 4px; }

/* ── Divider ── */
hr { border-color: rgba(201,162,39,0.2) !important; }

/* ── Checkbox ── */
.stCheckbox label span { color: #e8e8e8 !important; }

</style>
""", unsafe_allow_html=True)

HP_CASTLE   = "https://upload.wikimedia.org/wikipedia/commons/c/c4/Wizarding_World_of_Harry_Potter_Castle.jpg"
HP_DIAGON   = "https://upload.wikimedia.org/wikipedia/commons/e/e1/Diagon_Alley%2C_The_making_of_Harry_Potter_%28Ank_Kumar%2C_Infosys%29_02.jpg"
HP_STUDIO   = "https://upload.wikimedia.org/wikipedia/commons/b/b5/Entrance_to_the_Making_of_Harry_Potter_studio_tour.jpg"
HP_PANORAMA = "https://upload.wikimedia.org/wikipedia/commons/2/25/The_Wizarding_World_of_Harry_Potter_-_panoramio_%281%29.jpg"
HP_WORDMARK = "https://upload.wikimedia.org/wikipedia/commons/6/6e/Harry_Potter_wordmark.svg"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Hogwarts castle in sidebar
    st.markdown(
        f"<img src='{HP_STUDIO}' style='width:100%;border-radius:10px;"
        f"border:1px solid rgba(201,162,39,0.3);margin-bottom:12px;"
        f"box-shadow:0 0 20px rgba(201,162,39,0.15)' />",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style='text-align:center; padding: 4px 0 16px'>
      <div style='font-family:Cinzel,serif; font-size:1.15rem; color:#c9a227;
                  text-shadow: 0 0 20px rgba(201,162,39,0.6); letter-spacing:3px;'>
        Wizarding Knowledge Graph
      </div>
      <div style='color:#666; font-size:0.72rem; margin-top:4px; letter-spacing:1px;'>
        BERT · REBEL · NetworkX
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Filter Entities**")
    entity_filter = st.multiselect(
        "", list(ENTITY_COLORS.keys()),
        default=list(ENTITY_COLORS.keys()),
        label_visibility="collapsed",
    )

    st.markdown("**Focus on Node**")
    focus_node  = st.text_input("", placeholder="e.g. harry_potter", label_visibility="collapsed")
    focus_depth = st.slider("Depth", 1, 3, 1)

    show_labels = st.checkbox("Show edge labels", value=True)

    st.divider()
    # Entity type legend
    st.markdown("**Legend**")
    for etype, color in ENTITY_COLORS.items():
        if etype == "Other": continue
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
            f"<span style='width:12px;height:12px;border-radius:50%;background:{color};"
            f"display:inline-block;box-shadow:0 0 6px {color}80'></span>"
            f"<span style='color:#ccc;font-size:0.83rem'>{etype}</span></div>",
            unsafe_allow_html=True,
        )

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div style='position:relative;border-radius:16px;overflow:hidden;margin-bottom:28px;
                box-shadow:0 0 60px rgba(201,162,39,0.25)'>
      <img src='{HP_CASTLE}' style='width:100%;height:280px;object-fit:cover;
           object-position:center 55%;display:block;filter:brightness(0.45)'/>
      <div style='position:absolute;inset:0;display:flex;flex-direction:column;
                  align-items:center;justify-content:center;
                  background:linear-gradient(to bottom,rgba(5,5,15,0.1),rgba(5,5,15,0.7))'>
        <img src='{HP_WORDMARK}' style='height:64px;filter:brightness(0) invert(1)
             sepia(1) saturate(4) hue-rotate(8deg) brightness(1.1);margin-bottom:10px'/>
        <div style='font-family:Cinzel,serif;font-size:1.1rem;color:rgba(255,255,255,0.7);
                    letter-spacing:4px;text-transform:uppercase'>Knowledge Graph</div>
        <div style='color:rgba(255,255,255,0.45);font-size:0.8rem;margin-top:8px;letter-spacing:1px'>
          BERT NER · REBEL Relation Extraction · NetworkX · 50 Wikipedia articles
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Pipeline runner ───────────────────────────────────────────────────────────

with st.expander("⚙️  Run / Refresh ML Pipeline", expanded=not os.path.exists(GRAPH_CACHE)):
    col_l, col_r = st.columns([3, 1])
    with col_l:
        use_rebel   = st.checkbox("Use REBEL for relation extraction (recommended — ~1.5 GB first download)", value=True)
        force_fresh = st.checkbox("Force full re-run (ignore cache)")
    with col_r:
        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if not ML_AVAILABLE:
        st.info("ML models not available in this environment — graph loads from pre-built cache below.")
    if (run_btn or not os.path.exists(GRAPH_CACHE)) and ML_AVAILABLE:
        if force_fresh:
            for p in [NER_CACHE, RE_CACHE, GRAPH_CACHE, os.path.join("data","wiki_cache.json")]:
                if os.path.exists(p): os.remove(p)
            st.cache_resource.clear()

        prog = st.progress(0, "Step 1 / 4 — Fetching HP Wikipedia articles…")
        corpus    = load_corpus(use_cache=not force_fresh)
        sentences = corpus_to_sentences(corpus)
        prog.progress(25, f"Step 2 / 4 — BERT NER on {len(sentences)} sentences…")

        if os.path.exists(NER_CACHE):
            with open(NER_CACHE) as f: enriched = json.load(f)
        else:
            enriched = batch_extract(sentences, batch_size=16)
            with open(NER_CACHE,"w") as f: json.dump(enriched, f)

        prog.progress(50, f"Step 3 / 4 — {'REBEL' if use_rebel else 'spaCy SVO'} relation extraction…")

        if os.path.exists(RE_CACHE):
            with open(RE_CACHE) as f: triplets = json.load(f)
        else:
            triplets = extract_from_corpus(sentences, use_rebel=use_rebel)
            with open(RE_CACHE,"w") as f: json.dump(triplets, f)

        prog.progress(75, f"Step 4 / 4 — Building graph from {len(triplets)} triplets…")
        G_built = build_from_triplets(triplets)
        save_graph(G_built)
        st.cache_resource.clear()
        prog.progress(100, "Done!")
        st.success(f"Pipeline complete — **{G_built.number_of_nodes()} nodes**, **{G_built.number_of_edges()} edges** extracted by ML.")

# ── Load graph ────────────────────────────────────────────────────────────────

G_full = load_graph()
if G_full is None or G_full.number_of_nodes() == 0:
    st.info("No graph found — expand the pipeline above and click **Run Pipeline**.")
    st.stop()

# Filters
kept = [n for n, d in G_full.nodes(data=True) if d.get("entity_type", "Other") in entity_filter]
G = G_full.subgraph(kept).copy()
if focus_node.strip() and focus_node.strip() in G.nodes:
    G = get_subgraph(G, focus_node.strip(), depth=focus_depth)

# ── Stats row ─────────────────────────────────────────────────────────────────

stats = graph_stats(G)
cols  = st.columns(len(stats))
icons = {"nodes":"🔵","edges":"🔗","Character":"🧙","Location":"🏰","Spell":"✨","Object":"🪄","Event":"⚡","Organization":"🦅","House":"🏠","Other":"⬡","Sport":"🏆"}
for col, (k, v) in zip(cols, stats.items()):
    col.metric(f"{icons.get(k,'•')} {k.title()}", v)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["🕸️  Knowledge Graph", "🤖  BERT NER", "🔗  REBEL Triplets", "📊  Analytics", "📋  Tables"])

# ══ Tab 1 — Knowledge Graph ═══════════════════════════════════════════════════

with tabs[0]:
    # ── Usage guide ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:rgba(8,8,22,0.7);border:1px solid rgba(201,162,39,0.18);
         border-radius:12px;padding:14px 20px;margin-bottom:14px;
         font-family:Inter,"Segoe UI",sans-serif;'>
      <div style='color:#c9a227;font-weight:700;font-size:0.82rem;
                  text-transform:uppercase;letter-spacing:1.2px;margin-bottom:10px'>
        How to use this graph
      </div>
      <div style='display:flex;flex-wrap:wrap;gap:8px 20px;color:#aaa;font-size:0.82rem;line-height:1.6'>
        <span><b style='color:#e8e8e8'>🖱 Hover</b> a dot — see the name &amp; type</span>
        <span><b style='color:#e8e8e8'>🖱 Click</b> a dot — light up its connections</span>
        <span><b style='color:#e8e8e8'>🖱 Click canvas</b> — reset all highlights</span>
        <span><b style='color:#e8e8e8'>🖱 Scroll</b> — zoom in / out</span>
        <span><b style='color:#e8e8e8'>🖱 Drag</b> — pan around</span>
        <span><b style='color:#e8e8e8'>Fit button</b> — zoom to fit everything</span>
        <span><b style='color:#e8e8e8'>Sidebar → Filter</b> — show / hide entity types</span>
        <span><b style='color:#e8e8e8'>Sidebar → Focus on Node</b> — type e.g. <code style='color:#4ecdc4;background:rgba(78,205,196,0.1);padding:1px 5px;border-radius:3px'>harry_potter</code> to zoom into one character</span>
        <span><b style='color:#e8e8e8'>Depth slider</b> — expand how many hops you see around a focused node</span>
      </div>
      <div style='margin-top:10px;color:#666;font-size:0.76rem'>
        Node size = number of connections. Bigger dot = more central character/place.
        Edges were extracted by REBEL (BART transformer) from 50 real Wikipedia articles — no hardcoded data.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if G.number_of_nodes() == 0:
        st.warning("No nodes match current filters.")
    else:
        components.html(cytoscape_html(G, height=740), height=750, scrolling=False)

    # Relationship labels for top edges
    if G.number_of_edges() > 0:
        top_edges = sorted(G.edges(data=True), key=lambda e: -e[2].get("weight", 1))[:8]
        st.markdown(
            "<div style='display:flex;flex-wrap:wrap;gap:7px;margin-top:12px'>"
            + "".join(
                f"<span style='background:rgba(201,162,39,0.08);border:1px solid rgba(201,162,39,0.22);"
                f"border-radius:20px;padding:3px 12px;font-size:0.76rem;color:#c9a227'>"
                f"{G.nodes[u].get('label',u)} → {d.get('relation','?').replace('_',' ')} → {G.nodes[v].get('label',v)}"
                f"</span>"
                for u, v, d in top_edges
            )
            + "</div>",
            unsafe_allow_html=True,
        )

# ══ Tab 2 — BERT NER ══════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("""
    <div style='background:rgba(13,13,43,0.6);border:1px solid rgba(201,162,39,0.2);
         border-radius:12px;padding:16px 20px;margin-bottom:20px'>
      <b style='color:#c9a227'>Model:</b>
      <code style='color:#4ecdc4'>dslim/bert-base-NER</code> —
      BERT fine-tuned on CoNLL-2003.
      Extracts <b>PER</b> (people), <b>LOC</b> (locations),
      <b>ORG</b> (organisations), <b>MISC</b> from raw text.
    </div>
    """, unsafe_allow_html=True)

    demo_text = st.text_area(
        "Paste any Harry Potter text",
        value=(
            "Harry Potter and Hermione Granger studied at Hogwarts School of Witchcraft and Wizardry "
            "under Albus Dumbledore. Severus Snape taught Potions while secretly protecting Harry. "
            "Lord Voldemort was defeated at the Battle of Hogwarts when Neville Longbottom "
            "destroyed the final Horcrux."
        ),
        height=120,
    )

    if not ML_AVAILABLE:
        st.info("Live NER demo requires torch — not available in this deployment. Browse the corpus-wide results below.")
    if st.button("Extract Entities", key="ner_btn", disabled=not ML_AVAILABLE):
        with st.spinner("Running BERT NER…"):
            ents = extract_entities(demo_text)

        if ents:
            # Highlighted text
            _TYPE_BG = {"PER": ("#ff6b6b", "#1a0505"), "LOC": ("#4ecdc4", "#051a1a"),
                        "ORG": ("#c39bd3", "#150518"), "MISC": ("#ffe66d", "#1a1905")}
            highlighted = demo_text
            # Sort by start pos descending to avoid index shift
            for ent in sorted(ents, key=lambda e: -e["start"]):
                color, bg = _TYPE_BG.get(ent["entity_group"], ("#888", "#111"))
                span = (
                    f"<mark style='background:{bg};border:1px solid {color}60;"
                    f"border-radius:4px;padding:1px 5px;color:{color};font-weight:500'>"
                    f"{ent['word']}"
                    f"<sup style='font-size:0.65em;margin-left:3px;opacity:0.8'>{ent['entity_group']}</sup>"
                    f"</mark>"
                )
                highlighted = highlighted[:ent["start"]] + span + highlighted[ent["end"]:]
            st.markdown(
                f"<div style='background:#0d0d2b;border:1px solid rgba(201,162,39,0.2);"
                f"border-radius:10px;padding:16px;line-height:2;font-size:0.95rem'>{highlighted}</div>",
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns([2, 1])
            with col_a:
                df_e = pd.DataFrame(ents)[["word", "entity_group", "score"]]
                df_e.columns = ["Entity", "Type", "Confidence"]
                fig_e = px.bar(
                    df_e.head(20), x="Confidence", y="Entity", orientation="h",
                    color="Type",
                    color_discrete_map={"PER":"#ff6b6b","LOC":"#4ecdc4","ORG":"#c39bd3","MISC":"#ffe66d"},
                    title="Entities by Confidence Score",
                )
                fig_e.update_layout(
                    plot_bgcolor="#05050f", paper_bgcolor="#05050f",
                    font=dict(color="#c9a227"), yaxis=dict(categoryorder="total ascending"),
                    height=380,
                )
                st.plotly_chart(fig_e, use_container_width=True)
            with col_b:
                st.dataframe(df_e, use_container_width=True, hide_index=True)
        else:
            st.info("No entities found.")

    # Corpus stats
    if os.path.exists(NER_CACHE):
        st.divider()
        st.markdown("**Corpus-wide NER results**")
        with open(NER_CACHE) as f: ner_data = json.load(f)
        all_ents = [e for s in ner_data for e in s["entities"]]
        if all_ents:
            df_all  = pd.DataFrame(all_ents)
            c1, c2 = st.columns(2)
            with c1:
                tc = df_all["entity_group"].value_counts().reset_index()
                tc.columns = ["Type","Count"]
                fig_t = px.pie(tc, names="Type", values="Count", hole=0.5,
                    color="Type",
                    color_discrete_map={"PER":"#ff6b6b","LOC":"#4ecdc4","ORG":"#c39bd3","MISC":"#ffe66d"},
                    title="Entity Type Distribution")
                fig_t.update_layout(plot_bgcolor="#05050f",paper_bgcolor="#05050f",font=dict(color="#c9a227"))
                st.plotly_chart(fig_t, use_container_width=True)
            with c2:
                top20 = (df_all.groupby("word")["score"].agg(["count","mean"])
                         .sort_values("count",ascending=False).head(20).reset_index())
                top20.columns = ["Entity","Mentions","Avg Conf"]
                fig20 = px.bar(top20, x="Mentions", y="Entity", orientation="h",
                               title="Top 20 Most-Mentioned Entities",
                               color="Mentions", color_continuous_scale=["#1a1a3e","#c9a227"])
                fig20.update_layout(plot_bgcolor="#05050f",paper_bgcolor="#05050f",
                    font=dict(color="#c9a227"),yaxis=dict(categoryorder="total ascending"),height=500)
                st.plotly_chart(fig20, use_container_width=True)

# ══ Tab 3 — REBEL Triplets ════════════════════════════════════════════════════

with tabs[2]:
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.markdown(
            f"<img src='{HP_PANORAMA}' style='width:100%;border-radius:10px;"
            f"border:1px solid rgba(201,162,39,0.3);box-shadow:0 0 20px rgba(201,162,39,0.1)'/>",
            unsafe_allow_html=True,
        )
        st.caption("The Wizarding World of Harry Potter")
    with col_info:
        st.markdown("""
        <div style='background:rgba(13,13,43,0.6);border:1px solid rgba(201,162,39,0.2);
             border-radius:12px;padding:16px 20px;height:100%'>
          <div style='font-size:1.05rem;font-weight:600;color:#c9a227;margin-bottom:8px'>
            REBEL Relation Extraction
          </div>
          <code style='color:#4ecdc4'>Babelscape/rebel-large</code>
          <div style='color:#aaa;font-size:0.88rem;margin-top:10px;line-height:1.7'>
            BART-based seq2seq transformer. Reads raw text and generates structured
            <b style='color:#ffe66d'>(head, relation, tail)</b> triplets directly —
            no separate NER or classification step needed.<br><br>
            <b style='color:#c9a227'>Every edge in this graph</b> was extracted by this model
            from real Wikipedia text about Harry Potter.
          </div>
        </div>
        """, unsafe_allow_html=True)

    re_demo = st.text_area(
        "Test REBEL live",
        value=(
            "Harry Potter attended Hogwarts where Albus Dumbledore was headmaster. "
            "Hermione Granger used a Time-Turner to travel back in time and save Sirius Black. "
            "Severus Snape was a member of the Order of the Phoenix and taught Potions at Hogwarts."
        ),
        height=110,
        key="re_area",
    )
    if not ML_AVAILABLE:
        st.info("Live REBEL demo requires torch — not available in this deployment. Browse the pre-extracted triplets below.")
    if st.button("Extract Triplets", key="re_btn", disabled=not ML_AVAILABLE):
        with st.spinner("Running REBEL — downloads ~1.5 GB on first run…"):
            try:
                triplets = extract_triplets_rebel(re_demo)
                if triplets:
                    st.success(f"Extracted **{len(triplets)}** triplet(s)")
                    rows = []
                    for t in triplets:
                        rows.append({
                            "From (head)": t["head"],
                            "Relation":    t["relation"],
                            "To (tail)":   t["tail"],
                        })
                    df_t = pd.DataFrame(rows)
                    # Styled table with arrows
                    table_html = "<table style='width:100%;border-collapse:collapse'><thead><tr>" + \
                        "".join(f"<th style='text-align:left;color:#c9a227;padding:8px 12px;border-bottom:1px solid rgba(201,162,39,0.3)'>{c}</th>" for c in df_t.columns) + \
                        "</tr></thead><tbody>"
                    for _, row in df_t.iterrows():
                        table_html += (
                            f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05)'>"
                            f"<td style='padding:8px 12px;color:#ff6b6b'>{row['From (head)']}</td>"
                            f"<td style='padding:8px 12px;color:#888;font-style:italic'>→ {row['Relation']} →</td>"
                            f"<td style='padding:8px 12px;color:#4ecdc4'>{row['To (tail)']}</td>"
                            f"</tr>"
                        )
                    table_html += "</tbody></table>"
                    st.markdown(f"<div style='background:#0d0d2b;border:1px solid rgba(201,162,39,0.2);border-radius:10px;padding:8px'>{table_html}</div>", unsafe_allow_html=True)
                else:
                    st.warning("No triplets extracted from this text.")
            except Exception as e:
                st.error(f"REBEL error: {e}")

    if os.path.exists(RE_CACHE):
        st.divider()
        st.markdown("**All triplets extracted from corpus**")
        with open(RE_CACHE) as f: all_re = json.load(f)
        df_re = pd.DataFrame(all_re)
        if not df_re.empty and "relation" in df_re.columns:
            rel_counts = df_re["relation"].value_counts().head(20).reset_index()
            rel_counts.columns = ["Relation","Count"]
            fig_rel = px.bar(
                rel_counts, x="Count", y="Relation", orientation="h",
                title="Top 20 Relation Types Extracted by REBEL",
                color="Count", color_continuous_scale=["#1a1a3e","#c9a227"],
            )
            fig_rel.update_layout(plot_bgcolor="#05050f",paper_bgcolor="#05050f",
                font=dict(color="#c9a227"),yaxis=dict(categoryorder="total ascending"),height=500)
            st.plotly_chart(fig_rel, use_container_width=True)
            st.dataframe(df_re[["head","relation","tail","source"]].head(200),
                         use_container_width=True, hide_index=True)

# ══ Tab 4 — Analytics ═════════════════════════════════════════════════════════

with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        type_counts = {}
        for _, d in G.nodes(data=True):
            t = d.get("entity_type","Other"); type_counts[t] = type_counts.get(t,0) + 1
        fig_pie = px.pie(
            names=list(type_counts.keys()), values=list(type_counts.values()),
            color=list(type_counts.keys()), color_discrete_map=ENTITY_COLORS,
            title="Entity Type Distribution", hole=0.45,
        )
        fig_pie.update_traces(textfont_color="white")
        fig_pie.update_layout(plot_bgcolor="#05050f",paper_bgcolor="#05050f",font=dict(color="#c9a227"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        deg_data = [{"Entity":G.nodes[n].get("label",n),"Type":G.nodes[n].get("entity_type","?"),"Degree":G.degree(n)} for n in G.nodes]
        df_deg = pd.DataFrame(deg_data).sort_values("Degree",ascending=False).head(20)
        fig_deg = px.bar(
            df_deg, x="Degree", y="Entity", orientation="h",
            color="Type", color_discrete_map=ENTITY_COLORS,
            title="Top 20 Most-Connected Entities",
        )
        fig_deg.update_layout(plot_bgcolor="#05050f",paper_bgcolor="#05050f",
            font=dict(color="#c9a227"),yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_deg, use_container_width=True)

    st.markdown("**PageRank Centrality** — most important nodes by link structure")
    pr = nx.pagerank(G, alpha=0.85)
    df_pr = pd.DataFrame([
        {"Entity":G.nodes[n].get("label",n),"Type":G.nodes[n].get("entity_type","?"),"PageRank":round(v,5)}
        for n,v in pr.items()
    ]).sort_values("PageRank",ascending=False).head(15)
    fig_pr = px.bar(
        df_pr, x="PageRank", y="Entity", orientation="h",
        color="Type", color_discrete_map=ENTITY_COLORS,
        title="Top 15 Nodes by PageRank",
    )
    fig_pr.update_layout(plot_bgcolor="#05050f",paper_bgcolor="#05050f",
        font=dict(color="#c9a227"),yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_pr, use_container_width=True)

# ══ Tab 5 — Tables ════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("**Nodes**")
    node_rows = [{"ID":n,"Label":d.get("label",n),"Type":d.get("entity_type","?"),
                  "In":G.in_degree(n),"Out":G.out_degree(n),"Total":G.degree(n)}
                 for n,d in G.nodes(data=True)]
    st.dataframe(pd.DataFrame(node_rows).sort_values("Total",ascending=False),
                 use_container_width=True, hide_index=True)

    st.markdown("**Edges** (ML-extracted relationships)")
    edge_rows = [{"From":G.nodes[u].get("label",u),
                  "Relation":d.get("relation","?").replace("_"," "),
                  "To":G.nodes[v].get("label",v),
                  "Weight":d.get("weight",1),
                  "Source":", ".join(list(d.get("sources",[]))[:2])}
                 for u,v,d in G.edges(data=True)]
    st.dataframe(pd.DataFrame(edge_rows).sort_values("Weight",ascending=False),
                 use_container_width=True, hide_index=True)
