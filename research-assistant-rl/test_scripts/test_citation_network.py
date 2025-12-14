# tests/test_citation_network.py
from src.citation_network import build_citation_graph

def test_citation_graph_builds_internal_edges():
    papers = [
        {"title": "A", "url": "https://openalex.org/W1", "year": 2020, "citationCount": 10,
         "references": ["https://openalex.org/W2"]},
        {"title": "B", "url": "https://openalex.org/W2", "year": 2019, "citationCount": 50,
         "references": []},
        {"title": "C", "url": "https://openalex.org/W3", "year": 2021, "citationCount": 5,
         "references": ["https://openalex.org/W999"]},  # external, should not create edge
    ]

    G, stats = build_citation_graph(papers)

    assert stats.nodes == 3
    assert stats.edges == 1
    assert G.has_edge("https://openalex.org/W1", "https://openalex.org/W2")

def test_hubs_ranked_by_in_degree():
    papers = [
        {"title": "A", "url": "https://openalex.org/W1", "citationCount": 1, "references": ["https://openalex.org/W2"]},
        {"title": "B", "url": "https://openalex.org/W2", "citationCount": 99, "references": []},
        {"title": "C", "url": "https://openalex.org/W3", "citationCount": 3, "references": ["https://openalex.org/W2"]},
    ]
    G, stats = build_citation_graph(papers)
    assert stats.hubs[0]["url"] == "https://openalex.org/W2"
    assert stats.hubs[0]["in_degree"] == 2
