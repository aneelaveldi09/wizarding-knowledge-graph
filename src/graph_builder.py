"""Builds the Harry Potter knowledge graph using NetworkX."""

import networkx as nx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.hp_knowledge import (
    CHARACTERS, SPELLS, LOCATIONS, EVENTS, OBJECTS, RELATIONSHIPS, ENTITY_COLORS
)


def build_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    all_entities = CHARACTERS + SPELLS + LOCATIONS + EVENTS + OBJECTS
    for entity in all_entities:
        G.add_node(
            entity["id"],
            label=entity["label"],
            entity_type=entity["type"],
            color=ENTITY_COLORS.get(entity["type"], "#CCCCCC"),
            **{k: v for k, v in entity.items() if k not in ("id", "label", "type")},
        )

    for source, relation, target in RELATIONSHIPS:
        if G.has_node(source) and G.has_node(target):
            G.add_edge(source, target, relation=relation)

    return G


def get_subgraph(G: nx.DiGraph, node_id: str, depth: int = 1) -> nx.DiGraph:
    """Return an ego-graph around a node up to `depth` hops."""
    nodes = set()
    frontier = {node_id}
    for _ in range(depth):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.successors(n))
            next_frontier.update(G.predecessors(n))
        nodes.update(frontier)
        frontier = next_frontier - nodes
    nodes.update(frontier)
    return G.subgraph(nodes).copy()


def graph_stats(G: nx.DiGraph) -> dict:
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "characters": sum(1 for _, d in G.nodes(data=True) if d.get("entity_type") == "Character"),
        "spells": sum(1 for _, d in G.nodes(data=True) if d.get("entity_type") == "Spell"),
        "locations": sum(1 for _, d in G.nodes(data=True) if d.get("entity_type") == "Location"),
        "events": sum(1 for _, d in G.nodes(data=True) if d.get("entity_type") == "Event"),
        "objects": sum(1 for _, d in G.nodes(data=True) if d.get("entity_type") == "Object"),
    }
