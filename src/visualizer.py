"""
Graph visualisation — Obsidian-style Cytoscape.js.
  - cose physics layout (built-in, no CDN plugin needed)
  - Tiny glowing dots sized by degree
  - Labels hidden by default, fade in on hover / click
  - Edges: near-invisible hairlines, glow on selection
  - Click a node to illuminate neighbourhood; click canvas to reset
"""

from __future__ import annotations
import json
import networkx as nx

ENTITY_COLORS = {
    "Character":    "#e05c5c",
    "Location":     "#3ec9c0",
    "Spell":        "#f0d060",
    "Object":       "#72c4a0",
    "Event":        "#e07878",
    "Organization": "#a87ec8",
    "House":        "#f0b830",
    "Other":        "#556688",
}


def cytoscape_html(G: nx.DiGraph, height: int = 740) -> str:
    if G.number_of_nodes() == 0:
        return "<div style='color:#555;text-align:center;padding:60px;font-family:Inter,sans-serif'>No data.</div>"

    elements = []
    for node_id, data in G.nodes(data=True):
        etype  = data.get("entity_type", "Other")
        color  = ENTITY_COLORS.get(etype, "#556688")
        degree = G.degree(node_id)
        size   = max(10, min(40, 10 + degree * 2.2))
        elements.append({
            "group": "nodes",
            "data": {
                "id":     node_id,
                "label":  data.get("label", node_id),
                "type":   etype,
                "color":  color,
                "size":   size,
                "degree": degree,
            },
        })

    for u, v, data in G.edges(data=True):
        rel    = data.get("relation", "").replace("_", " ")
        weight = data.get("weight", 1)
        src_color = ENTITY_COLORS.get(
            G.nodes[u].get("entity_type", "Other"), "#556688"
        )
        elements.append({
            "group": "edges",
            "data": {
                "id":     f"{u}__{v}",
                "source": u,
                "target": v,
                "label":  rel,
                "weight": max(0.5, min(float(weight), 3.0)),
                "color":  src_color,
            },
        })

    legend_html = "".join(
        f'<div class="li"><div class="ld" style="background:{c};box-shadow:0 0 5px {c}99"></div>'
        f'<span>{t}</span></div>'
        for t, c in ENTITY_COLORS.items() if t != "Other"
    )

    elements_json = json.dumps(elements)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{width:100%;height:{height}px;overflow:hidden;background:transparent}}
#cy{{
  width:100%;height:{height}px;
  background:radial-gradient(ellipse at 40% 40%,#0c0c18 0%,#070710 50%,#04040c 100%);
  border-radius:12px;
  border:1px solid rgba(201,162,39,0.12);
}}
#wrap{{position:relative;width:100%;height:{height}px}}
#legend{{
  position:absolute;top:12px;left:12px;
  background:rgba(4,4,12,0.82);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:10px;padding:10px 13px;
  font-family:Inter,'Segoe UI',sans-serif;font-size:10.5px;color:#aaa;
  z-index:10;backdrop-filter:blur(8px);max-width:130px;
}}
#legend .title{{color:rgba(201,162,39,0.85);font-size:10px;font-weight:700;
  letter-spacing:1.2px;text-transform:uppercase;margin-bottom:7px}}
.li{{display:flex;align-items:center;gap:7px;margin:3px 0}}
.ld{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
#tip{{
  position:absolute;pointer-events:none;
  background:rgba(8,8,22,0.95);
  border-radius:10px;padding:10px 14px;
  font-family:Inter,'Segoe UI',sans-serif;font-size:12px;color:#ddd;
  display:none;z-index:20;max-width:210px;
  box-shadow:0 6px 30px rgba(0,0,0,0.7);
  border:1px solid rgba(255,255,255,0.07);
}}
#ctrls{{position:absolute;bottom:12px;right:12px;display:flex;gap:5px;z-index:10}}
.btn{{
  background:rgba(8,8,22,0.82);
  border:1px solid rgba(255,255,255,0.1);
  color:rgba(201,162,39,0.8);
  border-radius:7px;padding:5px 12px;font-size:11px;cursor:pointer;
  font-family:Inter,sans-serif;transition:all 0.15s;backdrop-filter:blur(6px);
}}
.btn:hover{{background:rgba(201,162,39,0.12);border-color:rgba(201,162,39,0.35)}}
#nodecount{{
  position:absolute;bottom:12px;left:12px;
  color:rgba(255,255,255,0.2);font-size:10px;
  font-family:Inter,sans-serif;z-index:10;
}}
</style>
</head>
<body>
<div id="wrap">
  <div id="cy"></div>
  <div id="legend"><div class="title">Types</div>{legend_html}</div>
  <div id="tip"></div>
  <div id="nodecount">{G.number_of_nodes()} nodes &middot; {G.number_of_edges()} edges</div>
  <div id="ctrls">
    <button class="btn" onclick="cy.fit(undefined,60)">Fit</button>
    <button class="btn" onclick="reset()">Reset</button>
  </div>
</div>
<script>
var elements={elements_json};

var cy=cytoscape({{
  container:document.getElementById('cy'),
  elements:elements,
  style:[
    {{
      selector:'node',
      style:{{
        'background-color':'data(color)',
        'width':'data(size)','height':'data(size)',
        'label':'',
        'border-width':0,
        'shadow-blur':12,
        'shadow-color':'data(color)',
        'shadow-opacity':0.5,
        'shadow-offset-x':0,'shadow-offset-y':0,
        'transition-property':'shadow-opacity,width,height,border-width,opacity',
        'transition-duration':'0.2s',
        'z-index':5,
      }}
    }},
    {{
      selector:'node.hover',
      style:{{
        'label':'data(label)',
        'font-size':11,
        'font-family':'Inter,"Segoe UI",Arial,sans-serif',
        'color':'#ffffff',
        'text-outline-color':'#000',
        'text-outline-width':1.2,
        'text-valign':'bottom','text-halign':'center','text-margin-y':5,
        'shadow-opacity':0.9,'shadow-blur':22,
        'z-index':50,
      }}
    }},
    {{
      selector:'node.selected',
      style:{{
        'label':'data(label)',
        'font-size':12,
        'font-family':'Inter,"Segoe UI",Arial,sans-serif',
        'color':'#ffffff',
        'text-outline-color':'#000',
        'text-outline-width':1.5,
        'text-valign':'bottom','text-halign':'center','text-margin-y':5,
        'border-width':2,'border-color':'data(color)',
        'shadow-opacity':1,'shadow-blur':30,
        'z-index':100,
      }}
    }},
    {{
      selector:'node.neighbour',
      style:{{
        'label':'data(label)',
        'font-size':10,
        'font-family':'Inter,"Segoe UI",Arial,sans-serif',
        'color':'rgba(255,255,255,0.75)',
        'text-outline-color':'#000',
        'text-outline-width':1,
        'text-valign':'bottom','text-halign':'center','text-margin-y':4,
        'shadow-opacity':0.7,
        'z-index':30,
      }}
    }},
    {{selector:'node.dimmed',style:{{'opacity':0.07}}}},
    {{
      selector:'edge',
      style:{{
        'width':0.7,
        'line-color':'rgba(100,110,170,0.18)',
        'target-arrow-color':'rgba(100,110,170,0.22)',
        'target-arrow-shape':'triangle',
        'arrow-scale':0.5,
        'curve-style':'bezier',
        'label':'',
        'z-index':1,
        'transition-property':'opacity,line-color,width',
        'transition-duration':'0.2s',
      }}
    }},
    {{
      selector:'edge.lit',
      style:{{
        'line-color':'data(color)',
        'target-arrow-color':'data(color)',
        'label':'data(label)',
        'font-size':9,'font-family':'Inter,sans-serif',
        'color':'rgba(200,200,200,0.6)',
        'text-background-color':'rgba(4,4,12,0.85)',
        'text-background-opacity':1,
        'text-background-padding':'2px',
        'edge-text-rotation':'autorotate',
        'width':1.5,'z-index':20,
      }}
    }},
    {{selector:'edge.dimmed',style:{{'opacity':0.02}}}},
  ],
  layout:{{
    name:'cose',
    idealEdgeLength:130,
    nodeOverlap:20,
    refresh:20,
    fit:true,
    padding:50,
    randomize:true,
    componentSpacing:60,
    nodeRepulsion:function(){{return 600000;}},
    edgeElasticity:function(){{return 90;}},
    nestingFactor:5,
    gravity:60,
    numIter:1400,
    initialTemp:200,
    coolingFactor:0.95,
    minTemp:1.0,
    animationDuration:900,
    animateFilter:function(){{return true;}},
  }},
  wheelSensitivity:0.2,
  minZoom:0.05,
  maxZoom:5,
}});

var tip=document.getElementById('tip');

cy.on('mouseover','node',function(evt){{
  var n=evt.target;
  if(!n.hasClass('selected'))n.addClass('hover');
  var d=n.data(),pos=evt.renderedPosition;
  tip.style.display='block';
  tip.style.left=(pos.x+18)+'px';
  tip.style.top=(pos.y-12)+'px';
  tip.innerHTML=
    '<div style="color:'+d.color+';font-weight:700;font-size:13px;margin-bottom:3px">'+d.label+'</div>'+
    '<div style="color:#666;font-size:9.5px;text-transform:uppercase;letter-spacing:1px">'+d.type+'</div>'+
    '<div style="margin-top:5px;color:#888;font-size:11px">'+d.degree+' connection'+(d.degree!==1?'s':'')+'</div>';
  tip.style.borderLeft='2px solid '+d.color;
}});
cy.on('mouseout','node',function(evt){{
  evt.target.removeClass('hover');tip.style.display='none';
}});
cy.on('mousemove','node',function(evt){{
  var p=evt.renderedPosition;
  tip.style.left=(p.x+18)+'px';tip.style.top=(p.y-12)+'px';
}});
cy.on('mouseover','edge',function(evt){{
  var d=evt.target.data(),p=evt.renderedPosition;
  if(!d.label)return;
  tip.style.display='block';
  tip.style.left=(p.x+12)+'px';tip.style.top=(p.y-10)+'px';
  tip.innerHTML='<span style="color:rgba(201,162,39,0.9);font-style:italic">'+d.label+'</span>';
  tip.style.borderLeft='2px solid rgba(201,162,39,0.6)';
}});
cy.on('mouseout','edge',function(){{tip.style.display='none';}});

cy.on('tap','node',function(evt){{
  var n=evt.target;
  cy.elements().removeClass('selected neighbour hover dimmed');
  cy.edges().removeClass('lit');
  var nb=n.neighborhood();
  cy.elements().not(n).not(nb).addClass('dimmed');
  n.addClass('selected');
  nb.nodes().addClass('neighbour');
  nb.edges().addClass('lit');
  cy.edges().not(nb.edges()).addClass('dimmed');
}});
cy.on('tap',function(evt){{if(evt.target===cy)reset();}});

function reset(){{
  cy.elements().removeClass('selected neighbour hover dimmed');
  cy.edges().removeClass('lit');
  tip.style.display='none';
}}
</script>
</body>
</html>"""
