import requests
import arxiv
import time
import hashlib
import json
import os
from typing import List, Dict, Optional

from src.utils import rate_limit


class CachedAPI:
    """Base class with caching functionality"""

    CACHE_VERSION = "v2"  # bump when schema/behavior changes

    def __init__(self):
        self.cache_dir = "results/cache/api"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_key(self, query: str, source: str, limit: int) -> str:
        """
        Cache key must include limit.
        Otherwise changing slider won't change results (it will keep returning cached list).
        """
        key_str = f"{self.CACHE_VERSION}_{source}_{query}_limit={int(limit)}".encode("utf-8")
        return hashlib.md5(key_str).hexdigest()

    def get_cached(self, cache_key: str) -> Optional[List[Dict]]:
        """Retrieve from cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_cache(self, cache_key: str, data: List[Dict]) -> None:
        """Save to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass


class OpenAlexAPI(CachedAPI):
    """
    OpenAlex API client.

    Notes:
    - No API key required.
    - Supports referenced_works which enables citation-network edges.
    """

    def __init__(self):
        super().__init__()
        self.base_url = "https://api.openalex.org/works"
        self.session = requests.Session()
        # Polite pool: identify yourself (replace with your email)
        self.session.headers.update({"User-Agent": "mailto:your.email@example.com"})

    @rate_limit(max_per_minute=100)
    def get_paper_with_references(self, paper_id: str) -> Dict:
        """
        Fetch full paper details including references.

        Args:
            paper_id: OpenAlex ID (e.g., W2345678) or full URL (https://openalex.org/W2345678)

        Returns:
            Minimal paper dict with references
        """
        if not paper_id:
            return {}

        # Extract ID from URL if needed
        if "openalex.org" in paper_id:
            paper_id = paper_id.rstrip("/").split("/")[-1]

        url = f"https://api.openalex.org/works/{paper_id}"

        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return {}

            data = response.json()
            references = list(data.get("referenced_works", []) or [])

            return {
                "id": data.get("id", ""),
                "title": data.get("title", ""),
                "references": references,
                "year": data.get("publication_year", 0),
                "citationCount": data.get("cited_by_count", 0),
                "url": data.get("id", ""),
                "source": "openalex",
            }

        except Exception as e:
            print(f"Error fetching paper details: {e}")
            return {}

    @rate_limit(max_per_minute=100)  # polite use
    def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search OpenAlex for papers.

        Args:
            query: Search query string
            limit: Max number of papers to return

        Returns:
            List of paper dictionaries (standard format)
        """
        limit = int(limit)

        cache_key = self.get_cache_key(query, "openalex", limit)
        cached = self.get_cached(cache_key)
        if cached:
            return cached

        params = {
            "search": query,
            "per_page": min(limit, 200),
            "sort": "cited_by_count:desc",
            # Ask OpenAlex for referenced_works so citation edges can be built
            "select": "id,title,publication_year,cited_by_count,authorships,abstract_inverted_index,referenced_works",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(self.base_url, params=params, timeout=15)

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])

                    papers: List[Dict] = []
                    for work in results:
                        # Reconstruct abstract from inverted index if present
                        abstract = None
                        inv_index = work.get("abstract_inverted_index")
                        if inv_index:
                            try:
                                max_pos = max(max(positions) for positions in inv_index.values())
                                words = [""] * (max_pos + 1)
                                for word, positions in inv_index.items():
                                    for pos in positions:
                                        if 0 <= pos < len(words):
                                            words[pos] = word
                                abstract = " ".join(words).strip()
                            except Exception:
                                abstract = None

                        papers.append(
                            {
                                "title": work.get("title", "No title"),
                                "abstract": abstract or "No abstract available",
                                "year": work.get("publication_year", 0),
                                "citationCount": work.get("cited_by_count", 0),
                                "authors": [
                                    {"name": a.get("author", {}).get("display_name", "Unknown")}
                                    for a in work.get("authorships", [])
                                ],
                                "url": work.get("id", ""),
                                "references": work.get("referenced_works", []) or [],
                                "source": "openalex",
                            }
                        )

                    # Cache result
                    self.save_cache(cache_key, papers)
                    return papers

                if response.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue

                if response.status_code >= 500:
                    time.sleep(2 * (attempt + 1))
                    continue

                print(f"OpenAlex error: {response.status_code}")
                return []

            except requests.exceptions.Timeout:
                time.sleep(2 * (attempt + 1))
                continue
            except Exception as e:
                print(f"OpenAlex error: {e}")
                time.sleep(2 * (attempt + 1))
                continue

        return []


class ArxivAPI(CachedAPI):
    """arXiv API client with caching"""

    def __init__(self):
        super().__init__()
        self.client = arxiv.Client()

    @rate_limit(max_per_minute=20)
    def search_papers(self, query: str, limit: int = 10) -> List[Dict]:
        """Search arXiv with caching"""
        limit = int(limit)

        cache_key = self.get_cache_key(query, "arxiv", limit)
        cached = self.get_cached(cache_key)
        if cached:
            return cached

        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: List[Dict] = []
        try:
            for result in self.client.results(search):
                papers.append(
                    {
                        "title": result.title,
                        "abstract": result.summary,
                        "year": result.published.year,
                        "authors": [{"name": a.name} for a in result.authors],
                        "url": result.entry_id,
                        "citationCount": 0,
                        "references": [],  # arXiv API does not provide citation graph
                        "source": "arxiv",
                    }
                )

            self.save_cache(cache_key, papers)

        except Exception as e:
            print(f"arXiv error: {e}")

        return papers


class ResearchToolkit:
    """
    Unified interface to research APIs.
    Uses OpenAlex + arXiv.
    """

    def __init__(self):
        self.openalex = OpenAlexAPI()
        self.arxiv = ArxivAPI()
        self.call_count = {"openalex": 0, "arxiv": 0}
        self.failure_count = {"openalex": 0, "arxiv": 0}

    def search(self, query: str, source: str, limit: int = 10) -> List[Dict]:
        """
        Search papers from specified source.

        Args:
            query: Search query string
            source: 'openalex' or 'arxiv'
            limit: Max papers to return

        Returns:
            List of paper dictionaries
        """
        source = (source or "").lower()
        limit = int(limit)

        self.call_count[source] = self.call_count.get(source, 0) + 1

        try:
            if source == "openalex":
                papers = self.openalex.search_papers(query, limit)
            elif source == "arxiv":
                papers = self.arxiv.search_papers(query, limit)
            else:
                print(f"Unknown source: {source}")
                papers = []

            if not papers:
                self.failure_count[source] = self.failure_count.get(source, 0) + 1

            return papers

        except Exception as e:
            print(f"Search error for {source}: {e}")
            self.failure_count[source] = self.failure_count.get(source, 0) + 1
            return []

    def get_stats(self) -> Dict:
        """Return API usage statistics"""
        return {
            "total_calls": sum(self.call_count.values()),
            "by_source": self.call_count.copy(),
            "failures": self.failure_count.copy(),
            "success_rate": {
                src: 1 - (self.failure_count.get(src, 0) / max(1, self.call_count.get(src, 1)))
                for src in self.call_count.keys()
            },
        }
