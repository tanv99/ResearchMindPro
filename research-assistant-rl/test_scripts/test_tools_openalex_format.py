# tests/test_tools_openalex_format.py
import types
from src.tools import OpenAlexAPI

class DummyResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

def test_openalex_parses_papers_and_references(monkeypatch):
    api = OpenAlexAPI()

    payload = {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "title": "Test Paper",
                "publication_year": 2022,
                "cited_by_count": 12,
                "authorships": [{"author": {"display_name": "Alice"}}],
                "abstract_inverted_index": None,
                "referenced_works": ["https://openalex.org/W999"]
            }
        ]
    }

    def fake_get(self, url, params=None, timeout=10):
        return DummyResp(200, payload)

    monkeypatch.setattr(api.session, "get", types.MethodType(fake_get, api.session))

    papers = api.search_papers("transformer", limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p["title"] == "Test Paper"
    assert p["year"] == 2022
    assert p["citationCount"] == 12
    assert p["url"] == "https://openalex.org/W123"
    assert isinstance(p["references"], list)
    assert p["references"][0] == "https://openalex.org/W999"
    assert p["authors"][0]["name"] == "Alice"
