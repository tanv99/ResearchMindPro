from collections import Counter
import re


class PaperSynthesizer:
    """Learns to extract and combine key information from papers"""
    
    def __init__(self):
        self.synthesis_history = []
        self.key_terms_learned = set()
    
    def extract_key_terms(self, papers):
        """Extract important terms from paper titles/abstracts"""
        text = ' '.join([
            f"{p.get('title', '')} {p.get('abstract', '')}"
            for p in papers
        ]).lower()
        
        # Remove common words
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 
            'this', 'that', 'these', 'those', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'can', 'such', 'which', 'who',
            'when', 'where', 'how', 'what', 'why'
        }
        
        # Extract words (simple tokenization)
        words = re.findall(r'\b[a-z]{4,}\b', text)
        words = [w for w in words if w not in stopwords]
        
        # Get most common terms
        term_counts = Counter(words)
        key_terms = [term for term, count in term_counts.most_common(10) if count > 1]
        
        return key_terms
    
    def synthesize(self, papers, query_terms):
        """
        Synthesize insights from multiple papers.
        Returns synthesis quality score and generated text.
        
        Args:
            papers: List of paper dictionaries
            query_terms: Original search terms
            
        Returns:
            Dictionary with synthesis text and quality metrics
        """
        if not papers or len(papers) < 2:
            return {
                'synthesis': 'Insufficient papers for synthesis',
                'quality': 0.0,
                'terms': [],
                'new_terms_discovered': 0
            }
        
        # Extract key terms from papers
        key_terms = self.extract_key_terms(papers)
        
        # Learn new terms
        new_terms = set(key_terms) - self.key_terms_learned
        self.key_terms_learned.update(new_terms)
        
        # Calculate synthesis quality
        query_set = set(t.lower() for t in query_terms)
        term_set = set(key_terms)
        relevance = len(query_set & term_set) / len(query_set) if query_set else 0
        
        # Quality improves with: relevance + novelty + breadth
        breadth = min(1.0, len(key_terms) / 10)
        novelty = len(new_terms) / max(1, len(key_terms))
        quality = (relevance * 0.5) + (breadth * 0.3) + (novelty * 0.2)
        
        # Generate synthesis text
        top_terms = key_terms[:5]
        if top_terms:
            synthesis = f"Papers focus on: {', '.join(top_terms)}. "
        else:
            synthesis = "Limited common themes identified. "
        
        # Add citation statistics
        total_citations = sum(p.get('citationCount', 0) for p in papers)
        avg_citations = total_citations / len(papers)
        synthesis += f"Average citations: {avg_citations:.0f}."
        
        result = {
            'synthesis': synthesis,
            'quality': quality,
            'terms': key_terms,
            'new_terms_discovered': len(new_terms)
        }
        
        self.synthesis_history.append(quality)
        return result
    
    def get_improvement(self):
        """Show synthesis quality improvement over time"""
        if len(self.synthesis_history) < 10:
            return 0.0
        
        early = sum(self.synthesis_history[:10]) / 10
        recent = sum(self.synthesis_history[-10:]) / 10
        return recent - early
