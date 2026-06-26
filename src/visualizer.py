"""
Graph visualisation.
Primary: cytoscape_html() — Cytoscape.js embedded via HTML component.
  - Force-directed layout (cose)
  - Shaped nodes by entity type
  - Glow shadows, click-to-highlight, edge labels, dark theme
  - Used by serious knowledge graph tools (bioinformatics, research)
"""

from __future__ import annotations
import json
import networkx as nx

ENTITY_COLORS = {
    "Character":    "#ff6b6b",
    "Location":     "#4ecdc4",
    "Spell":        "#ffe66d",
    "Object":       "#a8e6cf",
    "Event":        "#ff8b94",
    "Organization": "#c39bd3",
    "House":        "#f9ca24",
    "Other":        "#666888",
}

ENTITY_SHAPES = {
    "Character":    "ellipse",
    "Location":     "round-rectangle",
    "Spell":        "diamond",
    "Object":       "pentagon",
    "Event":        "star",
    "Organization": "hexagon",
    "House":        "barrel",
    "Other":        "ellipse",
}


def cytoscape_html(G: nx.DiGraph, height: int = 740) -> str:
    """
    Render the knowledge graph using Cytoscape.js (via CDN).
    Returns a self-contained HTML string for st.components.html().
    """
    if G.number_of_nodes() == 0:
        return "<div style='color:#888;text-align:center;padding:40px'>No data to display.</div>"

    elements = []
    for node_id, data in G.nodes(data=True):
        etype  = data.get("entity_type", "Other")
        color  = ENTITY_COLORS.get(etype, "#666888")
        degree = G.degree(node_id)
        size   = max(28, min(72, 28 + degree * 4))
        elements.append({
            "group": "nodes",
            "data": {
                "id": node_id,
                "label": data.get("label", node_id),
                "type": etype,
                "color": color,
                "size": size,
                "degree": degree,
            },
        })

    for u, v, data in G.edges(data=True):
        rel    = data.get("relation", "").replace("_", " ")
        weight = data.get("weight", 1)
        elements.append({
            "group": "edges",
            "data": {
                "id": f"{u}__{v}",
                "source": u,
                "target": v,
                "label": rel,
                "weight": max(1.0, min(float(weight), 5.0)),
            },
        })

    # Build per-type style rules for shape
    shape_rules = "\n".join(
        f"""        {{ selector: 'node[type="{etype}"]', style: {{ 'shape': '{shape}' }} }},"""
        for etype, shape in ENTITY_SHAPES.items()
    )

    elements_json = json.dumps(elements)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: 100%; height: {height}px; background: transparent; overflow: hidden; }}
  #cy {{
    width: 100%;
    height: {height}px;
    background: radial-gradient(ellipse at 30% 35%, #0d1433 0%, #07071a 55%, #05050f 100%);
    border-radius: 12px;
    border: 1px solid rgba(201,162,39,0.2);
  }}
  #legend {{
    position: absolute;
    top: 14px;
    left: 14px;
    background: rgba(7,7,26,0.88);
    border: 1px solid rgba(201,162,39,0.25);
    border-radius: 10px;
    padding: 10px 14px;
    font-family: 'Segoe UI', Inter, Arial, sans-serif;
    font-size: 11px;
    color: #ccc;
    z-index: 10;
    backdrop-filter: blur(6px);
  }}
  #legend .title {{
    color: #c9a227;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 7px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 7px; margin: 4px 0; }}
  .legend-dot {{ width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }}
  #tooltip {{
    position: absolute;
    pointer-events: none;
    background: #0d0d2b;
    border-radius: 9px;
    padding: 10px 14px;
    font-family: 'Segoe UI', Inter, Arial, sans-serif;
    font-size: 12px;
    color: #e8e8e8;
    display: none;
    z-index: 20;
    max-width: 220px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.6);
  }}
  #controls {{
    position: absolute;
    bottom: 14px;
    right: 14px;
    display: flex;
    gap: 6px;
    z-index: 10;
  }}
  .ctrl-btn {{
    background: rgba(13,13,43,0.85);
    border: 1px solid rgba(201,162,39,0.3);
    color: #c9a227;
    border-radius: 7px;
    padding: 5px 11px;
    font-size: 12px;
    cursor: pointer;
    font-family: 'Segoe UI', Arial, sans-serif;
    transition: all 0.15s;
  }}
  .ctrl-btn:hover {{ background: rgba(201,162,39,0.15); }}
</style>
</head>
<body>
<div style="position:relative;width:100%;height:{height}px">
  <div id="cy"></div>

  <div id="legend">
    <div class="title">Entity Types</div>
    {"".join(
        f'<div class="legend-item"><div class="legend-dot" style="background:{c};box-shadow:0 0 6px {c}99"></div><span>{t}</span></div>'
        for t, c in ENTITY_COLORS.items() if t != "Other"
    )}
  </div>

  <div id="tooltip"></div>

  <div id="controls">
    <button class="ctrl-btn" onclick="cy.fit()">Fit</button>
    <button class="ctrl-btn" onclick="resetHighlight()">Reset</button>
  </div>
</div>

<script>
var elements = {elements_json};

var cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: elements,
  style: [
    {{
      selector: 'node',
      style: {{
        'background-color': 'data(color)',
        'label': 'data(label)',
        'width': 'data(size)',
        'height': 'data(size)',
        'font-size': 10,
        'font-family': 'Segoe UI, Inter, Arial, sans-serif',
        'color': '#ffffff',
        'text-outline-color': '#000000',
        'text-outline-width': 1.5,
        'text-valign': 'bottom',
        'text-halign': 'center',
        'text-margin-y': 5,
        'border-width': 2,
        'border-color': 'rgba(255,255,255,0.25)',
        'shadow-blur': 18,
        'shadow-color': 'data(color)',
        'shadow-opacity': 0.65,
        'shadow-offset-x': 0,
        'shadow-offset-y': 0,
        'z-index': 10,
        'transition-property': 'border-color, border-width, shadow-opacity, opacity',
        'transition-duration': '0.15s',
      }}
    }},
    {shape_rules}
    {{
      selector: 'node:selected',
      style: {{
        'border-width': 4,
        'border-color': '#c9a227',
        'shadow-opacity': 1,
        'shadow-blur': 30,
        'z-index': 999,
      }}
    }},
    {{
      selector: 'node.highlighted',
      style: {{
        'border-width': 3,
        'border-color': '#c9a227',
        'shadow-opacity': 0.9,
        'z-index': 50,
      }}
    }},
    {{
      selector: 'node.dimmed',
      style: {{ 'opacity': 0.12 }}
    }},
    {{
      selector: 'edge',
      style: {{
        'width': 'data(weight)',
        'line-color': 'rgba(201,162,39,0.28)',
        'target-arrow-color': 'rgba(201,162,39,0.55)',
        'target-arrow-shape': 'triangle',
        'arrow-scale': 0.8,
        'curve-style': 'bezier',
        'label': 'data(label)',
        'font-size': 9,
        'font-family': 'Segoe UI, Inter, Arial, sans-serif',
        'color': 'rgba(180,180,180,0.55)',
        'text-background-color': '#07071a',
        'text-background-opacity': 0.75,
        'text-background-padding': '2px',
        'text-border-opacity': 0,
        'edge-text-rotation': 'autorotate',
        'z-index': 1,
        'transition-property': 'opacity, line-color',
        'transition-duration': '0.15s',
      }}
    }},
    {{
      selector: 'edge.highlighted',
      style: {{
        'line-color': 'rgba(201,162,39,0.8)',
        'target-arrow-color': '#c9a227',
        'z-index': 20,
        'width': 2.5,
      }}
    }},
    {{
      selector: 'edge.dimmed',
      style: {{ 'opacity': 0.04 }}
    }},
  ],
  layout: {{
    name: 'cose',
    idealEdgeLength: 160,
    nodeOverlap: 24,
    refresh: 20,
    fit: true,
    padding: 40,
    randomize: true,
    componentSpacing: 80,
    nodeRepulsion: function() {{ return 520000; }},
    edgeElasticity: function() {{ return 100; }},
    nestingFactor: 5,
    gravity: 70,
    numIter: 1200,
    initialTemp: 250,
    coolingFactor: 0.95,
    minTemp: 1.0,
    animationDuration: 800,
  }},
  wheelSensitivity: 0.25,
  minZoom: 0.1,
  maxZoom: 4,
}});

// ── Click to highlight neighbours ────────────────────────────────────────────
cy.on('tap', 'node', function(evt) {{
  var node = evt.target;
  cy.elements().addClass('dimmed').removeClass('highlighted');
  node.removeClass('dimmed').addClass('highlighted');
  node.neighborhood().removeClass('dimmed').addClass('highlighted');
  node.connectedEdges().removeClass('dimmed').addClass('highlighted');
}});

cy.on('tap', function(evt) {{
  if (evt.target === cy) resetHighlight();
}});

function resetHighlight() {{
  cy.elements().removeClass('dimmed highlighted');
}}

// ── Tooltip ──────────────────────────────────────────────────────────────────
var tooltip = document.getElementById('tooltip');

cy.on('mouseover', 'node', function(evt) {{
  var d = evt.target.data();
  var pos = evt.renderedPosition;
  tooltip.style.display = 'block';
  tooltip.style.left = (pos.x + 16) + 'px';
  tooltip.style.top  = (pos.y - 10) + 'px';
  tooltip.innerHTML =
    '<div style="color:' + d.color + ';font-weight:700;font-size:13px;margin-bottom:4px">' + d.label + '</div>' +
    '<div style="color:#999;font-size:10px;text-transform:uppercase;letter-spacing:1px">' + d.type + '</div>' +
    '<div style="margin-top:5px">Connections: <b style="color:#c9a227">' + d.degree + '</b></div>';
  tooltip.style.borderLeft = '3px solid ' + d.color;
}});

cy.on('mouseout', 'node', function() {{
  tooltip.style.display = 'none';
}});

cy.on('mousemove', 'node', function(evt) {{
  var pos = evt.renderedPosition;
  tooltip.style.left = (pos.x + 16) + 'px';
  tooltip.style.top  = (pos.y - 10) + 'px';
}});

cy.on('mouseover', 'edge', function(evt) {{
  var d = evt.target.data();
  var pos = evt.renderedPosition;
  if (!d.label) return;
  tooltip.style.display = 'block';
  tooltip.style.left = (pos.x + 12) + 'px';
  tooltip.style.top  = (pos.y - 10) + 'px';
  tooltip.innerHTML = '<span style="color:#c9a227;font-style:italic">' + d.label + '</span>';
  tooltip.style.borderLeft = '3px solid rgba(201,162,39,0.6)';
}});

cy.on('mouseout', 'edge', function() {{
  tooltip.style.display = 'none';
}});
</script>
</body>
</html>"""
