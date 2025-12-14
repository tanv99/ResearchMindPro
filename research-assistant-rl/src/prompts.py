"""
LLM Prompt Engineering with NVIDIA NIM.
"""

import os
import requests
from typing import List, Dict

class PromptEngineer:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.model = model or os.getenv("NVIDIA_CHAT_MODEL")
        
        # FIXED URL - removed duplicate /openai/v1
        self.chat_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("Missing NVIDIA_API_KEY")
        if not self.model:
            raise ValueError("Missing NVIDIA_CHAT_MODEL")
    
    def _format_papers(self, papers: List[Dict], max_papers: int = 6) -> str:
        """Format papers for context"""
        lines = []
        for i, p in enumerate(papers[:max_papers], 1):
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            year = p.get("year", "")
            citations = p.get("citationCount", 0)
            
            if abstract and len(abstract) > 500:
                abstract = abstract[:500] + "..."
            
            lines.append(
                f"[Paper {i}] {title} ({year}, {citations} citations)\n"
                f"Abstract: {abstract}"
            )
        return "\n\n".join(lines)
    
    def synthesize_literature_review(self, papers: List[Dict], query_text: str) -> str:
        """Generate synthesis using NVIDIA LLM"""
        if not papers:
            return "No papers retrieved for synthesis."
        
        context = self._format_papers(papers)
        
        system_msg = "You are a research assistant. Synthesize academic papers concisely and accurately."
        
        user_msg = f"""Research Question: {query_text}

Papers:
{context}

Provide a synthesis covering:
1. Key themes (3-4 main topics)
2. Main findings with evidence
3. Methodological approaches
4. Research gaps
5. How papers connect

Format: 300-400 words, cite as [Paper 1], [Paper 2], etc.

Synthesis:"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.6,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                self.chat_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"LLM Error ({response.status_code}): {response.text[:200]}"
        
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    def generate_research_questions(self, papers: List[Dict]) -> List[str]:
        """Generate follow-up questions"""
        context = self._format_papers(papers, max_papers=3)
        
        prompt = f"""Based on these papers, generate 5 research questions:

{context}

Format as numbered list (1. 2. 3. etc.)"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 400
        }
        
        try:
            response = requests.post(self.chat_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"]
                questions = [line.strip() for line in text.split('\n') 
                           if line.strip() and any(c.isdigit() for c in line[:3])]
                return questions[:5]
            else:
                return []
        
        except Exception:
            return []