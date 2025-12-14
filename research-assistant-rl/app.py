"""
ResearchMind Pro - Interactive Research Assistant
Run: streamlit run app.py
"""

import os
import sys
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
sys.path.insert(0, os.path.abspath("."))

from src.environment import ResearchEnvironment, ResearchTask
from src.coordinator import EnhancedCoordinator

# Citation network (import once. avoid importing inside tab)
from src.citation_network import build_citation_graph, plot_citation_graph

st.set_page_config(page_title="ResearchMind Pro", layout="wide")

# Dark theme styling
st.markdown(
    """
<style>
  .stApp { background: #0e1117; }
  header[data-testid="stHeader"] { background: transparent; display: none; }

  [data-testid="stSidebar"] { background: #1a1d29; border-right: 1px solid #2d3139; }
  [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }

  .main .block-container { padding-top: 2.25rem; background: transparent; }

  h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
  p, div, span, label { color: #c9d1d9 !important; }

  .stTextInput input, .stSelectbox select {
      background: #21262d !important;
      border: 1px solid #30363d !important;
      border-radius: 8px !important;
      color: #c9d1d9 !important;
  }

  .stButton button {
      background: #238636;
      color: white;
      border: none;
      border-radius: 8px;
      padding: 10px 14px;
      font-weight: 600;
  }
  .stButton button:hover { background: #2ea043; }

  [data-testid="stSidebar"] .stButton button {
      background: transparent;
      border: 1px solid #30363d;
      color: #c9d1d9;
      text-align: left;
      padding: 10px 14px;
      border-radius: 10px;
  }
  [data-testid="stSidebar"] .stButton button:hover {
      background: #21262d;
      border-color: #58a6ff;
  }

  [data-testid="stMetricValue"] { color: #58a6ff !important; }
  [data-testid="stMetricLabel"] { color: #8b949e !important; }

  .streamlit-expanderHeader {
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 10px;
      color: #c9d1d9 !important;
  }
  .streamlit-expanderContent {
      background: #0d1117;
      border: 1px solid #30363d;
  }

  .stJson { background: #161b22; border: 1px solid #30363d; border-radius: 10px; }

  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# Load config from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "researchmind-papers")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "")

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# Helper functions
def render_papers(papers):
    """Display retrieved papers"""
    if not papers:
        st.info("No papers returned.")
        return

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        url = paper.get("url", "")
        year = paper.get("year", "N/A")
        citations = paper.get("citationCount", 0)
        abstract = paper.get("abstract", "No abstract")

        with st.expander(f"{i}. {title}"):
            col_a, col_b = st.columns([3, 1])

            with col_a:
                st.markdown(f"**Abstract:** {abstract[:600]}{'...' if len(abstract) > 600 else ''}")
                if url:
                    st.markdown(f"[View Paper]({url})")

            with col_b:
                st.metric("Year", year)
                try:
                    st.metric("Citations", f"{int(citations):,}")
                except Exception:
                    st.metric("Citations", str(citations))


def render_rag_matches(matches):
    """Display RAG semantic search results"""
    if not matches:
        st.info("No Pinecone matches. Papers stored but may need different query.")
        return

    # De-dup matches to avoid repeated display
    seen = set()
    deduped = []
    for m in matches:
        key = (m.get("url", ""), m.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    for i, match in enumerate(deduped, 1):
        title = match.get("title", "Unknown")
        score = float(match.get("score", 0) or 0)
        year = match.get("year", "N/A")
        citations = match.get("citationCount", 0)
        abstract = match.get("abstract", "Not cached")

        with st.expander(f"{i}. {title} (Similarity: {score:.3f})"):
            try:
                st.markdown(f"**Year:** {year} | **Citations:** {int(citations):,}")
            except Exception:
                st.markdown(f"**Year:** {year} | **Citations:** {citations}")
            st.markdown(f"**Abstract:** {abstract[:400]}{'...' if len(abstract) > 400 else ''}")


# Sidebar Navigation
with st.sidebar:
    st.markdown("## Navigation")

    if st.button("Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"

    if st.button("Interactive Demo", use_container_width=True):
        st.session_state.page = "Interactive Demo"

    if st.button("Project Resources", use_container_width=True):
        st.session_state.page = "Project Resources"

    st.markdown("---")
    st.markdown("### Configuration")

    use_rag = st.checkbox("Enable RAG", value=True)
    use_llm = st.checkbox("Enable LLM", value=True)
    papers_to_retrieve = st.slider("Papers to retrieve", 5, 30, 15)


# Initialize
env = ResearchEnvironment()
coordinator = EnhancedCoordinator(
    use_rag=use_rag,
    use_llm=use_llm,
    pinecone_key=PINECONE_API_KEY if use_rag else None,
    pinecone_index=PINECONE_INDEX_NAME,
    nvidia_key=NVIDIA_API_KEY if use_llm else None,
    nvidia_chat_model=NVIDIA_CHAT_MODEL,
)

# Pages
page = st.session_state.page

if page == "Dashboard":
    st.title("ResearchMind Pro")
    st.markdown("Multi-Agent RL for Intelligent Research Discovery")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Overview")
        st.markdown(
            """
ResearchMind uses reinforcement learning to optimize research paper retrieval.

- Q-Learning learns query strategies
- UCB Bandit learns database selection
- Multi-agent coordination with voting
- RAG stores papers in Pinecone
- LLM generates literature reviews
- Citation network visualization
"""
        )

    with col2:
        st.markdown("### Performance")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Reward", "+37.5%")
            st.metric("Relevance", "+15.5%")
        with col_b:
            st.metric("P-value", "< 0.001")
            st.metric("Cohen's d", "0.94")


elif page == "Interactive Demo":
    st.title("Interactive Research Demo")
    st.markdown("Run research tasks and view RL decisions, papers, semantic search, AI synthesis, and citation networks.")

    col1, col2 = st.columns([2, 1])

    with col1:
        topic = st.selectbox(
            "Research Topic",
            ["machine_learning", "nlp", "computer_vision", "systems", "theory"],
        )

    with col2:
        difficulty = st.selectbox(
            "Task Difficulty",
            ["easy", "medium", "hard"],
            index=1,
        )

    query_text = st.text_input(
        "Research Query",
        value="transformer attention mechanism",
        placeholder="Enter search terms",
    )

    st.markdown("")

    if st.button("Run Research Task", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("Please enter a research query")
        else:
            terms = [t.strip() for t in query_text.split() if t.strip()]
            task = ResearchTask(topic=topic, query_terms=terms, difficulty=difficulty)
            env.current_task = task

            with st.spinner("RL agents optimizing search..."):
                papers, reward, meta = coordinator.research_with_fallback(
                    env, task, limit=papers_to_retrieve
                )

            papers = papers or []
            st.success(f"✓ Retrieved {len(papers)} papers")

            # Tabs for results
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                [
                    "Agent Decision",
                    "Retrieved Papers",
                    "AI Synthesis",
                    "Semantic Search (RAG)",
                    "Citation Network",
                ]
            )

            # TAB 1: Agent Decision
            with tab1:
                st.markdown("### RL Agent Decision")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Strategy", meta.get("strategy", "N/A"))
                c2.metric("Source", meta.get("source", "N/A"))
                c3.metric("Papers", len(papers))
                c4.metric("Reward", f"{float(reward):.2f}")

                with st.expander("View Full Decision Metadata"):
                    st.json(
                        {
                            "strategy": meta.get("strategy"),
                            "source": meta.get("source"),
                            "papers_retrieved": len(papers),
                            "reward": round(float(reward), 3),
                            "relevance": round(float(meta.get("relevance", 0) or 0), 3),
                            "fallback_used": meta.get("fallback_used"),
                            "allocation": meta.get("allocation"),
                            "rag_enabled": meta.get("rag_enabled"),
                            "llm_enabled": meta.get("llm_enabled"),
                            "rag_upserted": meta.get("rag_upserted", 0),
                            "requested_k": meta.get("requested_k", papers_to_retrieve),
                        }
                    )

                st.markdown("#### How Agent Decided")
                st.markdown(
                    f"""
- **Task Allocation:** `{meta.get('allocation', 'both')}`  
- **Query Strategy:** `{meta.get('strategy', 'unknown')}`  
- **Database:** `{meta.get('source', 'unknown')}`  
- **Fallback:** {"Used backup source" if meta.get("fallback_used") else "Primary source succeeded"}  
"""
                )

            # TAB 2: Retrieved Papers
            with tab2:
                st.markdown("### Retrieved Papers")
                st.caption(f"Showing {min(len(papers), 15)} of {len(papers)} papers")
                render_papers(papers[:15])

            # TAB 3: AI Synthesis
            with tab3:
                st.markdown("### AI-Generated Literature Review")

                if meta.get("llm_synthesis"):
                    synthesis = meta.get("llm_synthesis") or ""
                    if not synthesis.startswith("LLM Error"):
                        st.markdown(
                            f"""
<div style='background: rgba(88, 166, 255, 0.10);
            border-left: 4px solid #58a6ff;
            padding: 18px;
            border-radius: 10px;
            line-height: 1.7;'>
{synthesis}
</div>
""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(synthesis)
                else:
                    st.info("LLM synthesis not enabled. Enable it in the sidebar.")

            # TAB 4: RAG
            with tab4:
                st.markdown("### Pinecone Semantic Search Results")

                if meta.get("rag_matches"):
                    st.caption("Papers retrieved from Pinecone based on semantic similarity.")
                    render_rag_matches(meta.get("rag_matches", []))
                elif meta.get("rag_enabled"):
                    st.info(
                        f"Papers stored in Pinecone ({meta.get('rag_upserted', 0)} vectors). "
                        "Run another query to see semantic retrieval."
                    )
                else:
                    st.info("RAG not enabled. Enable it in the sidebar to use Pinecone.")

            # TAB 5: Citation Network
            with tab5:
                st.markdown("### Citation Network Analysis")

                if (meta.get("source") or "").lower() == "openalex" and len(papers) >= 5:
                    with st.spinner("Building citation network..."):
                        try:
                            G, stats = build_citation_graph(papers)

                            st.markdown("#### Network Statistics")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Papers (Nodes)", stats.nodes)
                            c2.metric("Internal Citations (Edges)", stats.edges)

                            # This will work only if stats has isolated_nodes
                            isolated = getattr(stats, "isolated_nodes", None)
                            if isolated is None:
                                c3.metric("Isolated Papers", "N/A")
                                st.caption("Note: isolated_nodes not provided. Update src/citation_network.py stats dataclass.")
                            else:
                                c3.metric("Isolated Papers", isolated)

                            st.markdown("")

                            if stats.hubs:
                                st.markdown("#### Most Influential Papers in This Set")
                                st.caption("Papers that are cited by other papers within your retrieved set.")
                                for i, hub in enumerate(stats.hubs[:3], 1):
                                    st.markdown(
                                        f"**{i}. {hub['title']}**  \n"
                                        f"Cited by {hub['in_degree']} papers (internal) | "
                                        f"{int(hub['citations']):,} total citations | "
                                        f"Year: {hub['year']}"
                                    )

                            if stats.edges > 0:
                                st.markdown("#### Interactive Citation Network")
                                fig = plot_citation_graph(G)
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption(
                                    "Tip: Hover over nodes for details. "
                                    "Node size = total citations."
                                )
                            else:
                                st.warning(
                                    "No internal citation connections found. "
                                    "Try increasing papers to 25-30 or using a more specific OpenAlex query."
                                )

                        except Exception as e:
                            st.error(f"Citation network error: {str(e)}")

                elif (meta.get("source") or "").lower() != "openalex":
                    st.info("Citation network is only available for OpenAlex results. arXiv does not provide references.")
                else:
                    st.info("Need at least 5 OpenAlex papers to build a citation network.")


elif page == "Project Resources":
    st.title("Project Resources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Reinforcement Learning")
        st.markdown(
            """
- Q-Learning for query strategy
- UCB Bandit for source selection
- Multi-agent coordination
- Task allocation by difficulty

**Results**
- +37.5% reward improvement
- p < 0.001 significance
- Cohen's d = 0.94
"""
        )

    with col2:
        st.markdown("### Features")
        st.markdown(
            """
- RAG with Pinecone vector storage
- LLM synthesis with NVIDIA
- Citation network visualization
- Semantic paper retrieval
- Error handling with fallbacks
"""
        )

    st.markdown("---")
    st.markdown("### Repository")
    st.markdown("[GitHub: github.com/tanv99/ResearchMindPro](https://github.com/tanv99/ResearchMindPro)")

    st.markdown("---")
    st.markdown("### Visualizations")

    if os.path.exists("results/learning_curves.png"):
        st.image("results/learning_curves.png", caption="Learning Curves")

    if os.path.exists("results/source_preferences.png"):
        st.image("results/source_preferences.png", caption="Source Preferences")


st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:13px;'>"
    "ResearchMind Pro . RL + RAG + Prompt Engineering + Citation Analysis"
    "</div>",
    unsafe_allow_html=True,
)
