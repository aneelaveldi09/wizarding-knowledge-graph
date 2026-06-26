"""
Builds NetworkX knowledge graph entirely from ML-extracted triplets.
No hardcoded relationships.
"""

from __future__ import annotations
import json
import os
import networkx as nx

from src.entity_normalizer import normalize_triplet, get_entity_type, get_entity_color

GRAPH_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "graph_cache.json")


def build_from_triplets(raw_triplets: list[dict]) -> nx.DiGraph:
    """Normalize triplets and construct the knowledge graph."""
    G = nx.DiGraph()

    for raw in raw_triplets:
        triplet = normalize_triplet(raw)
        if triplet is None:
            continue

        for node_id, label in [(triplet["head"], triplet["head_label"]),
                                (triplet["tail"], triplet["tail_label"])]:
            if not G.has_node(node_id):
                etype = get_entity_type(node_id)
                G.add_node(
                    node_id,
                    label=label.title() if node_id == triplet["head"] else triplet["tail_label"].title(),
                    entity_type=etype,
                    color=get_entity_color(node_id),
                )

        # Accumulate sources for duplicate edges
        relation = triplet["relation"]
        source = triplet["source"]
        if G.has_edge(triplet["head"], triplet["tail"]):
            G[triplet["head"]][triplet["tail"]]["weight"] += 1
            G[triplet["head"]][triplet["tail"]]["sources"].add(source)
        else:
            G.add_edge(
                triplet["head"],
                triplet["tail"],
                relation=relation,
                weight=1,
                sources={source},
            )

    # Convert sets to lists for JSON-serializability
    for u, v, d in G.edges(data=True):
        d["sources"] = list(d.get("sources", []))

    return G


def save_graph(G: nx.DiGraph, path: str = GRAPH_CACHE) -> None:
    data = nx.node_link_data(G, edges="edges")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str = GRAPH_CACHE) -> nx.DiGraph | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # Support both old ("links") and new ("edges") NetworkX serialization formats
    if "links" in data and "edges" not in data:
        data["edges"] = data.pop("links")
    return nx.node_link_graph(data, directed=True, edges="edges")


def graph_stats(G: nx.DiGraph) -> dict:
    type_counts: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        t = d.get("entity_type", "Other")
        type_counts[t] = type_counts.get(t, 0) + 1
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        **type_counts,
    }


def get_subgraph(G: nx.DiGraph, node_id: str, depth: int = 1) -> nx.DiGraph:
    nodes, frontier = set(), {node_id}
    for _ in range(depth):
        next_f = set()
        for n in frontier:
            next_f.update(G.successors(n))
            next_f.update(G.predecessors(n))
        nodes.update(frontier)
        frontier = next_f - nodes
    nodes.update(frontier)
    return G.subgraph(nodes).copy()
