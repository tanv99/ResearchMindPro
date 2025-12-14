from collections import Counter
from typing import Optional, Dict, Any, List, Tuple

from src.agents import QueryStrategyAgent, SourceSelectorAgent
from src.synthesis import PaperSynthesizer
from src.rag import RAGSystem
from src.prompts import PromptEngineer


class EnhancedCoordinator:
    def __init__(
        self,
        use_rag: bool = False,
        use_llm: bool = False,
        pinecone_key: Optional[str] = None,
        pinecone_index: str = "researchmind-papers",
        nvidia_key: Optional[str] = None,
        nvidia_embed_model: Optional[str] = None,  # kept for compatibility, may be unused
        nvidia_chat_model: Optional[str] = None,
    ):
        self.q_agent = QueryStrategyAgent()
        self.ucb_agent = SourceSelectorAgent()
        self.synthesizer = PaperSynthesizer()

        self.use_rag = use_rag
        self.use_llm = use_llm

        # RAG
        self.rag = None
        if use_rag and pinecone_key:
            try:
                self.rag = RAGSystem(
                    pinecone_api_key=pinecone_key,
                    index_name=pinecone_index,
                    max_local_papers=60,
                )
            except Exception as e:
                print(f"RAG init error: {e}")

        # LLM
        self.prompt_eng = None
        if use_llm and nvidia_key:
            try:
                self.prompt_eng = PromptEngineer(
                    api_key=nvidia_key,
                    model=nvidia_chat_model,
                )
            except Exception as e:
                print(f"LLM init error: {e}")

        self.task_allocation_history = {"q_agent": 0, "ucb_agent": 0, "both": 0}

    def allocate_task(self, task):
        if getattr(self.q_agent, "episode_count", 0) < 50:
            self.task_allocation_history["both"] += 1
            return "both"

        if task.difficulty == "easy":
            self.task_allocation_history["ucb_agent"] += 1
            return "ucb_agent"
        elif task.difficulty == "hard":
            self.task_allocation_history["q_agent"] += 1
            return "q_agent"

        self.task_allocation_history["both"] += 1
        return "both"

    def agent_voting(self, state, topic):
        votes = {}
        q_strategy, q_source = self.q_agent.choose_action(state)
        votes["q_agent"] = q_source

        ucb_source = self.ucb_agent.choose_source(topic)
        votes["ucb_agent"] = ucb_source

        vote_counts = Counter(votes.values())
        winning_source = vote_counts.most_common(1)[0][0]
        return winning_source, q_strategy, votes

    def research_with_fallback(self, env, task, limit: int = 10):
        """
        limit controls how many papers env.execute_search fetches from OpenAlex/arXiv.
        RAG top_k is capped by min(8, limit).
        """
        state = self.q_agent.get_state(task)
        allocation = self.allocate_task(task)

        if allocation == "ucb_agent":
            strategy = "specific"
            source = self.ucb_agent.choose_source(task.topic)
        elif allocation == "q_agent":
            strategy, source = self.q_agent.choose_action(state)
        else:
            source, strategy, _ = self.agent_voting(state, task.topic)

        papers = None
        sources_tried = [source]

        try:
            papers, cost = env.execute_search(strategy, source, limit=limit)
            if not papers:
                raise ValueError("No papers")
        except Exception:
            backup = "arxiv" if source == "openalex" else "openalex"
            sources_tried.append(backup)
            try:
                papers, cost = env.execute_search(strategy, backup, limit=limit)
                source = backup
            except Exception:
                papers, cost = [], 5.0

        if papers and source == "openalex":
            enriched_papers = []
            for paper in papers:
                # Fetch full details with references
                paper_id = paper.get('url', '')
                if paper_id:
                    try:
                        details = env.toolkit.openalex.get_paper_with_references(paper_id)
                        if details:
                            paper['references'] = details.get('references', [])
                    except Exception as e:
                        paper['references'] = []
                else:
                    paper['references'] = []
                
                enriched_papers.append(paper)
            
            papers = enriched_papers

        # Base synthesis
        synthesis_result = self.synthesizer.synthesize(papers, task.query_terms)
        query_text = " ".join(task.query_terms)

        # RAG
        rag_upserted = 0
        rag_matches = []
        rag_top_k = min(8, int(limit))

        if self.rag and papers:
            try:
                rag_upserted = self.rag.store_papers(papers, source, query_text)
                rag_matches = self.rag.query(query_text, top_k=rag_top_k)
            except Exception as e:
                print(f"RAG error: {e}")

        
 

        # LLM synthesis
        llm_synthesis = None
        if self.prompt_eng and papers:
            try:
                synthesis_papers = rag_matches if rag_matches else papers
                llm_synthesis = self.prompt_eng.synthesize_literature_review(
                    synthesis_papers,
                    query_text,
                )
            except Exception as e:
                llm_synthesis = f"LLM Error: {str(e)}"

        # Reward
        if papers:
            base_reward = env.get_reward(papers, cost)
            synthesis_bonus = float(synthesis_result.get("quality", 0) or 0) * 2
            total_reward = base_reward + synthesis_bonus
        else:
            total_reward = -10

        # Update agents
        next_state = state
        if allocation in ["q_agent", "both"]:
            self.q_agent.update(state, (strategy, source), total_reward, next_state)
        if allocation in ["ucb_agent", "both"]:
            self.ucb_agent.update(task.topic, source, total_reward)

        return papers, total_reward, {
            "strategy": strategy,
            "source": source,
            "cost": cost,
            "relevance": task.evaluate_results(papers) if papers else 0,
            "papers_count": len(papers),
            "synthesis": synthesis_result.get("synthesis", ""),
            "synthesis_quality": synthesis_result.get("quality", 0),
            "allocation": allocation,
            "sources_tried": sources_tried,
            "fallback_used": len(sources_tried) > 1,
            "rag_enabled": bool(self.rag),
            "rag_upserted": rag_upserted,
            "rag_matches": rag_matches,
            "rag_top_k": rag_top_k,
            "llm_enabled": bool(self.prompt_eng),
            "llm_synthesis": llm_synthesis,
            "requested_k": int(limit),
        }
