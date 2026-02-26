"""
streamlit_app.py — Production RAG Pipeline UI

Features:
  - Chat interface with conversation history
  - Sidebar configuration panel (change model, retrieval settings live)
  - Metrics dashboard (latency, sources, RAGAS scores)
  - Experiment comparison chart
  - Source document viewer
  - Hallucination warning banner
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yaml

# ─── Path Setup ───────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import RAGPipeline, RAGConfig, create_pipeline_from_yaml

# ─── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    border-left: 4px solid #7c3aed;
}
.source-card {
    background: #2d2d3f;
    border-radius: 8px;
    padding: 12px;
    margin: 6px 0;
    font-size: 0.85em;
}
.hallucination-banner {
    background: #7f1d1d;
    color: #fca5a5;
    padding: 12px;
    border-radius: 8px;
    border-left: 4px solid #ef4444;
}
.success-banner {
    background: #14532d;
    color: #86efac;
    padding: 12px;
    border-radius: 8px;
    border-left: 4px solid #22c55e;
}
.chat-message {
    padding: 12px 16px;
    border-radius: 10px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────
def init_session_state():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "total_latency" not in st.session_state:
        st.session_state.total_latency = 0.0
    if "config_changed" not in st.session_state:
        st.session_state.config_changed = False


# ─── Pipeline Loader ──────────────────────────────────
@st.cache_resource(show_spinner="🔧 Initializing RAG Pipeline...")
def load_pipeline(
    embedding_provider: str,
    llm_provider: str,
    retrieval_method: str,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    vector_store: str,
    query_expansion: bool,
    reranking: bool,
    force_rebuild: bool = False,
) -> Optional[RAGPipeline]:
    """Load and cache the RAG pipeline with given configuration."""
    try:
        config = RAGConfig(
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            retrieval_method=retrieval_method,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vector_store_provider=vector_store,
            query_expansion=query_expansion,
            reranking=reranking,
            force_rebuild=force_rebuild,
        )
        pipeline = RAGPipeline(config)
        pipeline.initialize()
        return pipeline
    except Exception as e:
        st.error(f"❌ Pipeline initialization failed: {e}")
        return None


# ─── Sidebar ──────────────────────────────────────────
def render_sidebar() -> Dict[str, Any]:
    """Render sidebar configuration panel."""
    st.sidebar.title("⚙️ Pipeline Configuration")

    st.sidebar.subheader("🤖 Models")
    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        ["openai", "ollama", "google"],
        help="openai=GPT-3.5, ollama=local Mistral, google=Gemini",
    )
    embedding_provider = st.sidebar.selectbox(
        "Embedding Model",
        ["huggingface", "openai"],
        help="huggingface=free (MiniLM), openai=paid (ada-002)",
    )

    st.sidebar.subheader("📚 Retrieval")
    retrieval_method = st.sidebar.radio("Method", ["similarity", "mmr"])
    top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=10, value=4)

    st.sidebar.subheader("✂️ Chunking")
    chunk_size = st.sidebar.select_slider(
        "Chunk Size (chars)", options=[128, 256, 512, 1024, 2048], value=512
    )
    chunk_overlap = st.sidebar.slider("Overlap (chars)", min_value=0, max_value=200, value=50, step=10)

    st.sidebar.subheader("🗄️ Vector Store")
    vector_store = st.sidebar.radio("Provider", ["chroma", "faiss"])

    st.sidebar.subheader("🚀 Advanced Features")
    query_expansion = st.sidebar.toggle("Query Expansion", value=False,
                                        help="Rewrite query using LLM before retrieval")
    reranking = st.sidebar.toggle("CrossEncoder Re-ranking", value=False,
                                   help="Re-rank retrieved docs with CrossEncoder")

    force_rebuild = st.sidebar.button("🔄 Rebuild Index", help="Force re-index all documents")

    st.sidebar.divider()
    st.sidebar.markdown("### 📂 Data")
    st.sidebar.info("Place PDF/TXT/DOCX files in `data/raw/` then rebuild index.")

    return {
        "llm_provider": llm_provider,
        "embedding_provider": embedding_provider,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "vector_store": vector_store,
        "query_expansion": query_expansion,
        "reranking": reranking,
        "force_rebuild": force_rebuild,
    }


# ─── Chat Interface ────────────────────────────────────
def render_chat(pipeline: Optional[RAGPipeline]):
    """Render the main chat interface."""
    st.subheader("💬 Ask Your Documents")

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander(f"📄 Sources ({len(msg['sources'])})", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Source {i}** — `{src['file']}`")
                        st.markdown(f"> {src['content'][:300]}...")

    # Query input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if pipeline is None:
            st.error("⚠️ Pipeline not initialized. Check configuration.")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving and generating..."):
                try:
                    response = pipeline.query(prompt)
                    st.session_state.last_response = response
                    st.session_state.query_count += 1
                    st.session_state.total_latency += response.latency_ms

                    # Display answer
                    st.markdown(response.answer)

                    # Source expander
                    sources = [
                        {
                            "file": doc.metadata.get("source_file", "Unknown"),
                            "content": doc.page_content,
                        }
                        for doc in response.source_documents
                    ]

                    if sources:
                        with st.expander(f"📄 Sources ({len(sources)})", expanded=False):
                            for i, src in enumerate(sources, 1):
                                st.markdown(f"**Source {i}** — `{src['file']}`")
                                st.code(src["content"][:400], language=None)

                    # Latency
                    st.caption(f"⚡ {response.latency_ms:.0f}ms | {len(sources)} sources retrieved")

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": sources,
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Clear conversation button
    if st.session_state.messages:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()


# ─── Metrics Panel ─────────────────────────────────────
def render_metrics_panel():
    """Render the metrics dashboard."""
    st.subheader("📊 Session Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Queries", st.session_state.query_count)

    with col2:
        avg_latency = (
            st.session_state.total_latency / st.session_state.query_count
            if st.session_state.query_count > 0 else 0
        )
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")

    with col3:
        if st.session_state.last_response:
            st.metric("Last Sources", len(st.session_state.last_response.source_documents))
        else:
            st.metric("Last Sources", "—")

    with col4:
        if st.session_state.last_response:
            ans_len = len(st.session_state.last_response.answer.split())
            st.metric("Answer Words", ans_len)
        else:
            st.metric("Answer Words", "—")


# ─── Experiment Results Dashboard ──────────────────────
def render_experiment_dashboard():
    """Render experiment comparison charts."""
    st.subheader("🧪 Experiment Results")

    results_dir = ROOT / "experiments" / "results"
    summary_path = results_dir / "experiment_summary.json"

    if not summary_path.exists():
        st.info("No experiment results found. Run `python experiments/experiment_runner.py --all` to generate results.")
        return

    with open(summary_path) as f:
        results = json.load(f)

    # Filter successful runs
    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        st.warning("No successful experiment runs found.")
        return

    df = pd.DataFrame([
        {
            "Run": r["run_name"],
            "Experiment": r["experiment"],
            "Value": r["param_value"],
            "Faithfulness": r.get("scores", {}).get("faithfulness"),
            "Answer Relevancy": r.get("scores", {}).get("answer_relevancy"),
            "Context Precision": r.get("scores", {}).get("context_precision"),
            "Context Recall": r.get("scores", {}).get("context_recall"),
            "Mean Score": r.get("mean_score", 0),
        }
        for r in successful
    ])

    # Experiment selector
    experiments = df["Experiment"].unique().tolist()
    selected_exp = st.selectbox("Select Experiment", experiments)
    filtered = df[df["Experiment"] == selected_exp]

    # Bar chart
    metrics_cols = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
    available_metrics = [m for m in metrics_cols if filtered[m].notna().any()]

    if available_metrics:
        fig = go.Figure()
        colors = ["#7c3aed", "#2563eb", "#16a34a", "#ea580c"]
        for i, metric in enumerate(available_metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=filtered["Value"],
                y=filtered[metric],
                marker_color=colors[i % len(colors)],
            ))

        fig.update_layout(
            title=f"{selected_exp} — RAGAS Metrics Comparison",
            xaxis_title=selected_exp.replace("_", " ").title(),
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            barmode="group",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart: baseline vs best
    if len(filtered) >= 2 and available_metrics:
        best_idx = filtered["Mean Score"].idxmax()
        base_idx = filtered.index[0]

        fig2 = go.Figure()
        for idx, label in [(base_idx, "Baseline"), (best_idx, "Best Config")]:
            row = filtered.loc[idx]
            values = [row.get(m, 0) or 0 for m in available_metrics]
            values += values[:1]  # close polygon

            fig2.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics + [available_metrics[0]],
                fill="toself",
                name=label,
            ))

        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Baseline vs Best Configuration",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Full results table
    st.dataframe(
        df.style.format({c: "{:.3f}" for c in metrics_cols if c in df.columns}),
        use_container_width=True,
    )


# ─── Source Viewer ─────────────────────────────────────
def render_source_viewer():
    """Display last retrieved sources in detail."""
    if st.session_state.last_response is None:
        st.info("Ask a question to see retrieved sources here.")
        return

    response = st.session_state.last_response
    st.subheader(f"📄 Retrieved Sources ({len(response.source_documents)})")

    for i, doc in enumerate(response.source_documents, 1):
        with st.expander(f"Source {i} — {doc.metadata.get('source_file', 'Unknown')}", expanded=(i == 1)):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(doc.page_content)
            with cols[1]:
                st.json({
                    k: v for k, v in doc.metadata.items()
                    if k in ("source_file", "page", "chunk_id", "chunk_size")
                })


# ─── Main App ──────────────────────────────────────────
def main():
    init_session_state()

    # Header
    st.title("🔍 Production RAG Pipeline")
    st.markdown("*Retrieval-Augmented Generation with RAGAS Evaluation & Experiment Tracking*")
    st.divider()

    # Sidebar config
    cfg = render_sidebar()

    # Load pipeline
    pipeline = load_pipeline(
        embedding_provider=cfg["embedding_provider"],
        llm_provider=cfg["llm_provider"],
        retrieval_method=cfg["retrieval_method"],
        top_k=cfg["top_k"],
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        vector_store=cfg["vector_store"],
        query_expansion=cfg["query_expansion"],
        reranking=cfg["reranking"],
        force_rebuild=cfg["force_rebuild"],
    )

    if pipeline and pipeline._initialized:
        st.success("✅ Pipeline ready")
    else:
        st.warning("⚠️ Pipeline not initialized — add documents to `data/raw/` and configure API keys.")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Metrics", "🧪 Experiments", "📄 Sources"])

    with tab1:
        render_chat(pipeline)

    with tab2:
        render_metrics_panel()
        if st.session_state.last_response:
            st.divider()
            st.subheader("Last Query Details")
            resp = st.session_state.last_response
            st.json({
                "question": resp.question,
                "expanded_question": resp.expanded_question,
                "latency_ms": resp.latency_ms,
                **resp.metadata,
            })

    with tab3:
        render_experiment_dashboard()

    with tab4:
        render_source_viewer()


if __name__ == "__main__":
    main()
