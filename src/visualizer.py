"""
Graph visualisation — two renderers:
  plotly_network()  → Plotly figure (inline, dark, glowing nodes)
  build_pyvis()     → Pyvis Network  (physics-based, draggable)
"""

from __future__ import annotations
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network

ENTITY_COLORS = {
    "Character":    "#ff6b6b",
    "Location":     "#4ecdc4",
    "Spell":        "#ffe66d",
    "Object":       "#a8e6cf",
    "Event":        "#ff8b94",
    "Organization": "#c39bd3",
    "House":        "#f9ca24",
    "Other":        "#888888",
}


# ── Plotly (primary, beautiful) ───────────────────────────────────────────────

def plotly_network(G: nx.DiGraph) -> go.Figure:
    if G.number_of_nodes() == 0:
        return go.Figure()

    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, k=2.5, seed=42, iterations=80)

    traces: list[go.BaseTraceType] = []

    # Edge lines
    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
    traces.append(go.Scatter(
        x=ex, y=ey, mode="lines", hoverinfo="none",
        line=dict(width=0.7, color="rgba(201,162,39,0.18)"),
        showlegend=False,
    ))

    # Nodes grouped by entity type
    for etype, color in ENTITY_COLORS.items():
        nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("entity_type", "Other") == etype]
        if not nodes:
            continue

        x  = [pos[n][0] for n, _ in nodes]
        y  = [pos[n][1] for n, _ in nodes]
        sz = [max(10, min(55, 10 + G.degree(n) * 4)) for n, _ in nodes]
        labels  = [d.get("label", n) for n, d in nodes]
        hover = [
            f"<b style='color:{color}'>{d.get('label', n)}</b><br>"
            f"<span style='color:#aaa'>Type: {etype}</span><br>"
            f"Connections: {G.degree(n)}  "
            f"(in {G.in_degree(n)} / out {G.out_degree(n)})"
            for n, d in nodes
        ]

        # Glow halo
        traces.append(go.Scatter(
            x=x, y=y, mode="markers", hoverinfo="none", showlegend=False,
            marker=dict(color=color, size=[s * 2.2 for s in sz], opacity=0.08, line=dict(width=0)),
        ))
        # Soft outer ring
        traces.append(go.Scatter(
            x=x, y=y, mode="markers", hoverinfo="none", showlegend=False,
            marker=dict(color=color, size=[s * 1.5 for s in sz], opacity=0.14, line=dict(width=0)),
        ))
        # Main node
        traces.append(go.Scatter(
            x=x, y=y,
            mode="markers+text",
            name=etype,
            text=labels,
            textposition="top center",
            textfont=dict(size=9, color="rgba(255,255,255,0.75)", family="Segoe UI, Arial"),
            marker=dict(
                color=color, size=sz, opacity=0.95,
                line=dict(width=1.5, color="rgba(255,255,255,0.35)"),
            ),
            hovertext=hover,
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor="#0d0d2b",
                bordercolor=color,
                font=dict(color="white", size=12, family="Segoe UI"),
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=730,
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="#05050f",
        paper_bgcolor="#05050f",
        font=dict(color="#c9a227", family="Segoe UI"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        legend=dict(
            bgcolor="rgba(10,10,30,0.85)",
            bordercolor="rgba(201,162,39,0.3)",
            borderwidth=1,
            font=dict(color="white", size=11),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


# ── Pyvis (physics mode, draggable) ──────────────────────────────────────────

_PYVIS_CSS = """
<style>
  html, body { margin:0; padding:0; background:#05050f; }
  #mynetwork {
    background: radial-gradient(ellipse at 35% 45%, #0d1433 0%, #05050f 75%) !important;
    border: 1px solid rgba(201,162,39,0.25) !important;
    border-radius: 12px !important;
  }
  div.vis-tooltip {
    background: #0d0d2b !important;
    border: 1px solid #c9a227 !important;
    color: #e8e8e8 !important;
    border-radius: 8px !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
    font-size: 13px !important;
    box-shadow: 0 0 24px rgba(201,162,39,0.3) !important;
    padding: 8px 12px !important;
  }
  div.vis-network canvas { filter: brightness(1.08) contrast(1.05); }
</style>
"""


def build_pyvis(G: nx.DiGraph, height: str = "700px") -> str:
    """Returns HTML string with custom-styled Pyvis graph."""
    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.set_options("""{
      "nodes": {
        "font": {"size": 13, "color": "#ffffff", "face": "Segoe UI"},
        "borderWidth": 2,
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.6)", "size": 12, "x": 0, "y": 0}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
        "color": {"color": "rgba(201,162,39,0.3)", "highlight": "#c9a227", "hover": "#ffd700"},
        "font": {"size": 9, "color": "rgba(200,200,200,0.7)", "align": "middle", "face": "Segoe UI"},
        "smooth": {"type": "cubicBezier", "roundness": 0.4},
        "selectionWidth": 2
      },
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -55,
          "centralGravity": 0.004,
          "springLength": 150,
          "springConstant": 0.07,
          "damping": 0.6
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 250, "updateInterval": 10}
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "tooltipDelay": 60,
        "multiselect": true
      }
    }""")

    for node_id, data in G.nodes(data=True):
        etype  = data.get("entity_type", "Other")
        color  = ENTITY_COLORS.get(etype, "#888")
        degree = G.degree(node_id)
        size   = max(14, min(52, 14 + degree * 3))
        net.add_node(
            node_id,
            label=data.get("label", node_id),
            color={"background": color, "border": _lighten(color), "highlight": {"background": _lighten(color), "border": "#fff"}, "hover": {"background": _lighten(color), "border": "#fff"}},
            size=size,
            title=_tooltip(node_id, data, G, color),
            shape=_shape(etype),
            borderWidth=2,
            borderWidthSelected=4,
        )

    for src, dst, data in G.edges(data=True):
        rel    = data.get("relation", "related_to").replace("_", " ")
        weight = data.get("weight", 1)
        net.add_edge(src, dst, title=rel, label=rel, width=max(1, min(weight * 0.8, 5)))

    html = net.generate_html()
    html = html.replace("</head>", _PYVIS_CSS + "\n</head>")
    return html


def _lighten(hex_color: str) -> str:
    """Return a slightly lighter version of a hex color for highlights."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r, g, b = min(255, r + 40), min(255, g + 40), min(255, b + 40)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color


def _shape(etype: str) -> str:
    return {"Character": "dot", "Location": "square", "Spell": "diamond",
            "Object": "triangle", "Event": "star", "Organization": "hexagon",
            "House": "ellipse"}.get(etype, "dot")


def _tooltip(node_id: str, data: dict, G: nx.DiGraph, color: str) -> str:
    label = data.get("label", node_id)
    etype = data.get("entity_type", "Unknown")
    return (
        f"<span style='color:{color};font-weight:bold;font-size:14px'>{label}</span><br>"
        f"<span style='color:#aaa'>Type: {etype}</span><br>"
        f"Connections: <b>{G.degree(node_id)}</b> &nbsp;"
        f"(in <b>{G.in_degree(node_id)}</b> / out <b>{G.out_degree(node_id)}</b>)"
    )
