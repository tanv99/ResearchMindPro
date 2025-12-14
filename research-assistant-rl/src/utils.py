import time
import json
import os
from functools import wraps


def rate_limit(max_per_minute):
    """
    Decorator to rate limit API calls.
    Ensures we don't exceed API limits.
    """
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator


def calculate_relevance_score(paper, query_terms):
    """
    Calculate how relevant a paper is to the search query.
    Handles None values from API responses.
    
    Args:
        paper: Paper dictionary with title/abstract
        query_terms: List of search terms
    
    Returns:
        Float score between 0-1
    """
    # Handle None values from API (FIX APPLIED)
    title = (paper.get('title') or '').lower()
    abstract = (paper.get('abstract') or '').lower()
    text = f"{title} {abstract}"
    
    if not text.strip():
        return 0.0
    
    query_terms = [t.lower() for t in query_terms]
    
    # Count term occurrences
    score = 0
    for term in query_terms:
        score += text.count(term)
    
    # Normalize by text length (per 1000 characters)
    text_length = max(len(text), 1)
    normalized_score = score / (text_length / 1000)
    
    # Cap at 1.0
    return min(1.0, normalized_score)


def save_cache(filename, data):
    """Save data to cache directory"""
    cache_dir = 'results/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, filename)
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Cache save failed: {e}")
        return False


def load_cache(filename):
    """Load data from cache if exists"""
    cache_path = os.path.join('results/cache', filename)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Cache load failed: {e}")
        return None
