"""
Citation network visualization with enhanced reference extraction.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import networkx as nx
import plotly.graph_objects as go


@dataclass
class NetworkStats:
    nodes: int
    edges: int
    hubs: List[Dict[str, Any]]
    isolated_nodes: int


def _extract_openalex_id(url_or_id: str) -> str:
    """Extract OpenAlex ID from URL or ID string"""
    if not url_or_id:
        return ""
    
    # If already just ID (W12345), return it
    if url_or_id.startswith('W'):
        return url_or_id
    
    # Extract from URL
    if 'openalex.org' in url_or_id:
        return url_or_id.split('/')[-1]
    
    return url_or_id


def build_citation_graph(papers: List[Dict]) -> Tuple[nx.DiGraph, NetworkStats]:
    """
    Build citation graph from papers.
    Edge A -> B means A cites B (references B).
    """
    G = nx.DiGraph()
    
    # Build ID to paper mapping
    id_to_paper = {}
    
    for paper in papers:
        paper_id = _extract_openalex_id(paper.get('url', ''))
        
        if not paper_id:
            continue
        
        id_to_paper[paper_id] = paper
        
        # Add node
        G.add_node(
            paper_id,
            title=paper.get('title', 'Untitled')[:80],  # Truncate for display
            year=paper.get('year', 0),
            citations=int(paper.get('citationCount', 0) or 0),
            source=paper.get('source', ''),
            url=paper.get('url', '')
        )
    
    # Add edges from references
    edges_added = 0
    for paper_id, paper in id_to_paper.items():
        references = paper.get('references', [])
        
        if not references:
            continue
        
        for ref in references:
            ref_id = _extract_openalex_id(ref)
            
            # Only add edge if both papers are in our set
            if ref_id and ref_id in id_to_paper:
                G.add_edge(paper_id, ref_id)
                edges_added += 1
    
    # Calculate hub papers (most cited within this set)
    in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    
    hubs = []
    for node_id, in_degree in in_degrees[:5]:
        if in_degree > 0:  # Only if actually cited
            node_data = G.nodes[node_id]
            hubs.append({
                'title': node_data.get('title', 'Unknown'),
                'in_degree': in_degree,
                'citations': node_data.get('citations', 0),
                'year': node_data.get('year', 0),
                'url': node_data.get('url', '')
            })
    
    # Count isolated nodes
    isolated = len([n for n in G.nodes() if G.degree(n) == 0])
    
    stats = NetworkStats(
        nodes=G.number_of_nodes(),
        edges=G.number_of_edges(),
        hubs=hubs,
        isolated_nodes=isolated
    )
    
    return G, stats


def plot_citation_graph(G: nx.DiGraph) -> go.Figure:
    """
    Create interactive Plotly visualization of citation network.
    """
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No papers to visualize",
            showarrow=False,
            font=dict(size=16, color="#8b949e")
        )
        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            height=400
        )
        return fig
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Create edges
    edge_traces = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.5, color='rgba(136, 146, 176, 0.3)'),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create nodes
    node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node data
        title = G.nodes[node].get('title', 'Unknown')
        year = G.nodes[node].get('year', 0)
        citations = G.nodes[node].get('citations', 0)
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        # Hover text
        node_text.append(
            f"<b>{title}</b><br>"
            f"Year: {year}<br>"
            f"Citations: {citations:,}<br>"
            f"Cited by (in set): {in_degree}<br>"
            f"References (in set): {out_degree}"
        )
        
        # Node size based on citations
        size = 15 + min(40, (citations ** 0.3))
        node_sizes.append(size)
        
        # Color based on in-degree (hub papers are red)
        if in_degree >= 3:
            node_colors.append('#ff6b6b')  # Red for hubs
        elif in_degree >= 1:
            node_colors.append('#ffd93d')  # Yellow for cited
        else:
            node_colors.append('#6bcf7f')  # Green for leaf nodes
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='#21262d')
        ),
        text=node_text,
        hoverinfo='text',
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title="Citation Network<br><sub>Node size = citations | Color: Red = hub, Yellow = cited, Green = leaf</sub>",
        showlegend=False,
        hovermode='closest',
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#c9d1d9"),
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig