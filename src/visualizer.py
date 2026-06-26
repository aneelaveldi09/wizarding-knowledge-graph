"""Generates interactive Pyvis graph HTML."""

from __future__ import annotations
import networkx as nx
from pyvis.network import Network


def build_pyvis(G: nx.DiGraph, height: str = "700px", physics: bool = True) -> Network:
    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.set_options("""
    {
      "nodes": {
        "font": {"size": 14, "color": "white"},
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
        "color": {"color": "#888888", "highlight": "#FFA500"},
        "font": {"size": 10, "align": "middle"},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.005,
          "springLength": 120,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "tooltipDelay": 100
      }
    }
    """)

    for node_id, data in G.nodes(data=True):
        label = data.get("label", node_id)
        color = data.get("color", "#CCCCCC")
        etype = data.get("entity_type", "Unknown")
        size = _node_size(G, node_id)
        title = _build_tooltip(node_id, data, G)
        net.add_node(
            node_id,
            label=label,
            color=color,
            size=size,
            title=title,
            shape=_node_shape(etype),
        )

    for src, dst, data in G.edges(data=True):
        relation = data.get("relation", "related_to")
        net.add_edge(src, dst, title=relation, label=relation)

    return net


def _node_size(G: nx.DiGraph, node_id: str) -> int:
    degree = G.degree(node_id)
    return max(15, min(50, 15 + degree * 2))


def _node_shape(entity_type: str) -> str:
    shapes = {
        "Character": "dot",
        "Spell": "diamond",
        "Location": "square",
        "Event": "star",
        "Object": "triangle",
    }
    return shapes.get(entity_type, "dot")


def _build_tooltip(node_id: str, data: dict, G: nx.DiGraph) -> str:
    lines = [f"<b>{data.get('label', node_id)}</b>", f"Type: {data.get('entity_type', 'Unknown')}"]
    if data.get("house"):
        lines.append(f"House: {data['house']}")
    if data.get("role"):
        lines.append(f"Role: {data['role']}")
    if data.get("effect"):
        lines.append(f"Effect: {data['effect']}")
    if data.get("category"):
        lines.append(f"Category: {data['category']}")
    out_edges = list(G.successors(node_id))
    in_edges = list(G.predecessors(node_id))
    lines.append(f"Connections: {len(out_edges) + len(in_edges)}")
    return "<br>".join(lines)


def render_to_file(G: nx.DiGraph, path: str) -> str:
    net = build_pyvis(G)
    net.save_graph(path)
    return path
