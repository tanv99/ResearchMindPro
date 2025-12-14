"""
RAG System using SentenceTransformers (local) + Pinecone.
Simpler and more reliable than NVIDIA embeddings.
"""

import os
import hashlib
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

class RAGSystem:
    """
    RAG with local embeddings (384 dim) + Pinecone storage.
    Keeps 60 papers locally, all in Pinecone.
    """
    
    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str = "researchmind-papers",
        max_local_papers: int = 60
    ):
        # Local embedding model (384 dimensions)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedding model loaded (384 dims)")
        
        # Local storage
        self.kb_dir = 'results/knowledge_base'
        os.makedirs(self.kb_dir, exist_ok=True)
        self.max_local_papers = max_local_papers
        
        # Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        
        try:
            self.index = self.pc.Index(index_name)
            stats = self.index.describe_index_stats()
            print(f"✓ Connected to Pinecone index: {index_name}")
            print(f"  Total vectors: {stats.get('total_vector_count', 0)}")
        except Exception as e:
            print(f"✗ Pinecone connection error: {e}")
            raise
    
    def generate_paper_id(self, paper: Dict) -> str:
        """Generate unique ID from title + source"""
        key = f"{paper.get('title', '')}_{paper.get('source', '')}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def embed_paper(self, paper: Dict) -> List[float]:
        """Generate embedding for paper"""
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        embedding = self.embedder.encode(text)
        return embedding.tolist()
    
    def manage_local_storage(self):
        """Keep only max_local_papers newest files"""
        import json
        
        paper_files = [f for f in os.listdir(self.kb_dir) if f.endswith('.json')]
        
        if len(paper_files) > self.max_local_papers:
            files_with_time = [
                (f, os.path.getmtime(os.path.join(self.kb_dir, f)))
                for f in paper_files
            ]
            files_with_time.sort(key=lambda x: x[1])
            
            num_to_delete = len(paper_files) - self.max_local_papers
            for filename, _ in files_with_time[:num_to_delete]:
                os.remove(os.path.join(self.kb_dir, filename))
    
    def store_papers(self, papers: List[Dict], source: str, query_text: str) -> int:
        """
        Store papers locally + upload to Pinecone.
        Returns number uploaded.
        """
        if not papers:
            return 0
        
        vectors = []
        
        for paper in papers:
            paper_id = self.generate_paper_id(paper)
            
            # Add metadata
            paper['paper_id'] = paper_id
            paper['source'] = source
            paper['retrieved_by_query'] = query_text
            
            # Generate embedding
            try:
                embedding = self.embed_paper(paper)
            except:
                continue
            
            # Pinecone metadata (size limited)
            metadata = {
                'title': paper.get('title', '')[:500],
                'year': paper.get('year', 0),
                'citations': paper.get('citationCount', 0),
                'source': source,
                'query': query_text[:200]
            }
            
            vectors.append((paper_id, embedding, metadata))
            
            # Store locally
            import json
            paper_file = os.path.join(self.kb_dir, f"{paper_id}.json")
            with open(paper_file, 'w', encoding='utf-8') as f:
                json.dump(paper, f, indent=2)
        
        # Upload to Pinecone
        if vectors:
            self.index.upsert(vectors=vectors)
            print(f"✓ Uploaded {len(vectors)} papers to Pinecone")
        
        # Clean local storage
        self.manage_local_storage()
        
        return len(vectors)
    
    def query(self, query_text: str, top_k: int = 8) -> List[Dict]:
        """
        Semantic search in Pinecone.
        Returns list of similar papers.
        """
        if not query_text:
            return []
        
        # Embed query
        query_embedding = self.embedder.encode(query_text).tolist()
        
        # Search Pinecone
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            print(f"Pinecone query error: {e}")
            return []
        
        # Format results
        matches = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            
            # Try loading full paper from local
            import json
            paper_id = match['id']
            paper_file = os.path.join(self.kb_dir, f"{paper_id}.json")
            
            if os.path.exists(paper_file):
                with open(paper_file, 'r', encoding='utf-8') as f:
                    paper = json.load(f)
                    paper['score'] = match['score']
                    paper['from_cache'] = True
                    matches.append(paper)
            else:
                # Use Pinecone metadata only
                matches.append({
                    'title': metadata.get('title', 'Unknown'),
                    'year': metadata.get('year', 0),
                    'citationCount': metadata.get('citations', 0),
                    'source': metadata.get('source', ''),
                    'abstract': '(Not in local cache)',
                    'score': match['score'],
                    'from_cache': False
                })
        
        return matches
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        import json
        
        local_files = [f for f in os.listdir(self.kb_dir) if f.endswith('.json')]
        local_size = sum(
            os.path.getsize(os.path.join(self.kb_dir, f))
            for f in local_files
        )
        
        try:
            pinecone_stats = self.index.describe_index_stats()
            pinecone_count = pinecone_stats.get('total_vector_count', 0)
        except:
            pinecone_count = 0
        
        return {
            'local_papers': len(local_files),
            'local_size_mb': local_size / (1024 * 1024),
            'local_limit': self.max_local_papers,
            'pinecone_papers': pinecone_count,
            'storage_location': os.path.abspath(self.kb_dir)
        }