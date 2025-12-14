import random
from typing import Dict, List, Tuple
from src.tools import ResearchToolkit
from src.utils import calculate_relevance_score


class ResearchTask:
    """Represents a research query task"""
    
    def __init__(self, topic: str, query_terms: List[str], difficulty: str):
        self.topic = topic
        self.query_terms = query_terms
        self.difficulty = difficulty
    
    def evaluate_results(self, papers: List[Dict]) -> float:
        """
        Score the quality of retrieved papers.
        Returns score between 0-1.
        """
        if not papers:
            return 0.0
        
        scores = []
        for paper in papers:
            relevance = calculate_relevance_score(paper, self.query_terms)
            
            # Boost score for highly-cited papers
            citations = paper.get('citationCount', 0)
            citation_boost = min(0.2, citations / 500)
            
            scores.append(relevance + citation_boost)
        
        return sum(scores) / len(scores)


class ResearchEnvironment:
    """Simulates research tasks for training RL agents"""
    
    def __init__(self):
        self.toolkit = ResearchToolkit()
        self.task_templates = self._create_task_templates()
        self.current_task = None
    
    def _create_task_templates(self) -> List[Dict]:
        """Define research scenarios across different topics"""
        return [
            {
                'topic': 'machine_learning',
                'queries': [
                    ['transformer', 'attention', 'mechanism'],
                    ['reinforcement', 'learning', 'deep'],
                    ['neural', 'architecture', 'search'],
                    ['meta', 'learning', 'few', 'shot']
                ]
            },
            {
                'topic': 'nlp',
                'queries': [
                    ['language', 'model', 'bert'],
                    ['sentiment', 'analysis', 'twitter'],
                    ['question', 'answering', 'squad'],
                    ['machine', 'translation', 'neural']
                ]
            },
            {
                'topic': 'computer_vision',
                'queries': [
                    ['object', 'detection', 'yolo'],
                    ['image', 'segmentation', 'semantic'],
                    ['face', 'recognition', 'deep'],
                    ['video', 'understanding', 'action']
                ]
            },
            {
                'topic': 'systems',
                'queries': [
                    ['distributed', 'systems', 'consensus'],
                    ['database', 'optimization', 'query'],
                    ['cloud', 'computing', 'scalability'],
                    ['operating', 'systems', 'scheduling']
                ]
            },
            {
                'topic': 'theory',
                'queries': [
                    ['algorithm', 'complexity', 'np'],
                    ['graph', 'algorithm', 'shortest', 'path'],
                    ['approximation', 'algorithm', 'optimization'],
                    ['online', 'learning', 'regret']
                ]
            }
        ]
    
    def generate_task(self) -> ResearchTask:
        """Generate a new research task"""
        template = random.choice(self.task_templates)
        query_terms = random.choice(template['queries'])
        difficulty = random.choice(['easy', 'medium', 'hard'])
        
        self.current_task = ResearchTask(
            topic=template['topic'],
            query_terms=query_terms,
            difficulty=difficulty
        )
        return self.current_task
    
    def execute_search(self, 
                      query_strategy: str, 
                      source: str, 
                      limit: int = 10) -> Tuple[List[Dict], float]:
        """
        Execute a search with the given strategy.
        
        Args:
            query_strategy: 'broad', 'specific', or 'narrow'
            source: 'openalex' or 'arxiv'
            limit: Number of papers to retrieve
        
        Returns:
            (papers, cost) where cost represents time/effort
        """
        if not self.current_task:
            raise ValueError("No active task")
        
        query = self._build_query(self.current_task.query_terms, query_strategy)
        papers = self.toolkit.search(query, source, limit)
        
        # Filter out papers with no content (FIX APPLIED)
        papers = [p for p in papers if p.get('title') or p.get('abstract')]
        
        # Calculate cost
        cost = len(papers) * 0.1
        
        if query_strategy == 'broad':
            cost *= 1.5
        elif query_strategy == 'narrow':
            cost *= 0.8
        
        return papers, cost
    
    def _build_query(self, terms: List[str], strategy: str) -> str:
        """Construct query string based on search strategy"""
        if strategy == 'broad':
            return ' OR '.join(terms)
        elif strategy == 'specific':
            return ' '.join(terms)
        elif strategy == 'narrow':
            return '"' + ' '.join(terms) + '"'
        
        return ' '.join(terms)
    
    def get_reward(self, papers: List[Dict], cost: float) -> float:
        """Calculate RL reward for search results"""
        relevance = self.current_task.evaluate_results(papers)
        reward = relevance * 10 - cost
        return reward
