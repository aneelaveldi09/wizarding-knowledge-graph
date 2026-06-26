"""Generates an interactive Pyvis graph from a NetworkX DiGraph."""

from __future__ import annotations
import networkx as nx
from pyvis.network import Network


def build_pyvis(G: nx.DiGraph, height: str = "680px", physics: bool = True) -> Network:
    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.set_options("""
    {
      "nodes": {
        "font": {"size": 13, "color": "white"},
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
        "color": {"color": "#666666", "highlight": "#FFA500"},
        "font": {"size": 9, "align": "middle", "color": "#cccccc"},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 140,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 200}
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "tooltipDelay": 80
      }
    }
    """)

    for node_id, data in G.nodes(data=True):
        label = data.get("label", node_id)
        color = data.get("color", "#AAAAAA")
        etype = data.get("entity_type", "Other")
        size = max(12, min(45, 12 + G.degree(node_id) * 3))
        tooltip = _tooltip(node_id, data, G)
        net.add_node(
            node_id,
            label=label,
            color=color,
            size=size,
            title=tooltip,
            shape=_shape(etype),
        )

    for src, dst, data in G.edges(data=True):
        relation = data.get("relation", "related_to").replace("_", " ")
        weight = data.get("weight", 1)
        net.add_edge(src, dst, title=relation, label=relation, width=min(weight, 5))

    return net


def _shape(etype: str) -> str:
    return {
        "Character": "dot",
        "Location": "square",
        "Spell": "diamond",
        "Object": "triangle",
        "Event": "star",
        "Organization": "hexagon",
        "House": "ellipse",
    }.get(etype, "dot")


def _tooltip(node_id: str, data: dict, G: nx.DiGraph) -> str:
    lines = [
        f"<b>{data.get('label', node_id)}</b>",
        f"Type: {data.get('entity_type', 'Unknown')}",
        f"Connections: {G.degree(node_id)}",
        f"In: {G.in_degree(node_id)}  |  Out: {G.out_degree(node_id)}",
    ]
    return "<br>".join(lines)
