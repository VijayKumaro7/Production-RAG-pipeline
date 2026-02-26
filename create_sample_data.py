"""
create_sample_data.py — Creates sample documents in data/raw/ for testing.

Run this if you don't have your own documents yet.
Generates 5 text files covering RAG-related topics.
"""

import os

SAMPLE_DOCS = {
    "01_rag_overview.txt": """
# Retrieval-Augmented Generation (RAG) — Overview

Retrieval-Augmented Generation (RAG) is a hybrid AI technique that combines a retrieval system with a language model to produce grounded, factual responses. Unlike pure language models that rely solely on parametric memory (weights learned during training), RAG systems actively fetch relevant information from an external knowledge base at inference time.

## How RAG Works

The RAG pipeline consists of two main phases:

1. **Retrieval Phase**: When a user submits a query, the system encodes it using an embedding model into a dense vector. This vector is compared against a pre-indexed corpus of document embeddings stored in a vector database. The top-K most similar document chunks are retrieved.

2. **Generation Phase**: The retrieved chunks are concatenated with the original query into a structured prompt. The language model then generates an answer conditioned on this context.

## Why RAG Matters

Pure language models suffer from several limitations:
- **Hallucination**: They may generate plausible-sounding but factually incorrect information
- **Knowledge cutoff**: Their knowledge is frozen at training time and cannot be updated
- **Opacity**: It's hard to trace which training data led to a given answer

RAG addresses all three: it grounds responses in documents, can be updated by adding new files to the corpus, and provides traceable source attribution.

## Applications

RAG is widely used in:
- Enterprise Q&A systems over internal documents
- Legal and medical document analysis
- Customer support automation
- Research literature review assistants
- Financial report analysis tools
""",

    "02_chunking_strategies.txt": """
# Document Chunking Strategies for RAG

Chunking is the process of splitting large documents into smaller pieces that can be efficiently embedded and retrieved. The choice of chunking strategy significantly affects RAG performance.

## Why Chunking Matters

Vector embedding models have a maximum token limit (typically 512 tokens for models like all-MiniLM-L6-v2). Entire documents cannot be embedded as a single unit. Additionally, retrieving smaller, focused chunks provides more precise context to the language model.

## Fixed-Size Chunking

The simplest approach: split documents every N characters regardless of content structure.

**Pros**: Simple, predictable chunk sizes, easy to implement
**Cons**: May split sentences or paragraphs mid-thought, losing semantic coherence

Best for: Uniform documents like log files or code

## Recursive Character Splitting

LangChain's RecursiveCharacterTextSplitter attempts to split on natural boundaries in order: paragraph breaks, line breaks, sentences, then words. This preserves semantic units better than fixed splitting.

**Pros**: Respects natural language structure, configurable separators
**Cons**: Chunk sizes may vary significantly

Best for: Mixed content documents, research papers, reports

## Sentence-Based Chunking

Uses NLP sentence detection (NLTK punkt tokenizer) to split exactly at sentence boundaries. Can be combined with a sliding window to group multiple sentences per chunk.

**Pros**: Maximally preserves semantic units
**Cons**: Requires NLTK, variable chunk sizes

Best for: Prose-heavy content like articles, books, documentation

## Chunk Overlap

Overlapping consecutive chunks by 10-20% of the chunk size ensures that information at chunk boundaries is not lost. A chunk_size=512 with chunk_overlap=50 means each chunk shares 50 characters with its neighbor.

## Optimal Parameters

Based on empirical testing:
- chunk_size: 512 characters provides a good balance between precision and context
- chunk_overlap: 50 characters (≈10%) prevents boundary information loss
- Recursive splitting outperforms fixed splitting on most document types
""",

    "03_vector_databases.txt": """
# Vector Databases for RAG Systems

Vector databases store and retrieve high-dimensional embedding vectors efficiently. Choosing the right vector store impacts retrieval speed, scalability, and feature support.

## ChromaDB

ChromaDB is an open-source, Python-native vector database designed for AI applications.

**Key features**:
- Persistent storage to disk (SQLite-backed)
- Metadata filtering alongside vector search
- Easy LangChain integration
- Supports cosine similarity, IP, and L2 distance metrics
- Free, runs locally, no external service needed

**Ideal for**: Development, small to medium corpora (up to millions of vectors), projects needing metadata filtering

**Limitations**: Not designed for distributed/production scale

## FAISS (Facebook AI Similarity Search)

FAISS is a high-performance library for dense vector similarity search, built by Meta AI Research.

**Key features**:
- Extremely fast approximate nearest neighbor (ANN) search
- Supports IndexFlatL2 (exact), IndexIVFFlat (approximate), and HNSW indexing
- Can handle billions of vectors
- GPU-accelerated variants available

**Ideal for**: High-throughput production systems, large-scale retrieval

**Limitations**: No built-in metadata filtering, requires manual index management

## Pinecone (Cloud)

A fully managed, cloud-native vector database with automatic scaling, real-time updates, and namespace partitioning.

**Ideal for**: Production workloads, teams without infrastructure expertise

**Limitations**: Paid service, vendor lock-in

## Retrieval Methods

### Cosine Similarity Search
Ranks results by cosine similarity between query and document vectors. Pure relevance ranking with no diversity mechanism.

### MMR (Maximal Marginal Relevance)
MMR balances relevance and diversity. After selecting the most relevant document, each subsequent selection maximizes relevance to the query while minimizing similarity to already-selected documents. This reduces redundancy in retrieved context.

The MMR score for candidate document d_i:
MMR(d_i) = λ × sim(q, d_i) - (1-λ) × max(sim(d_i, d_j))

Where λ controls the relevance-diversity tradeoff (0=max diversity, 1=max relevance).
""",

    "04_ragas_evaluation.txt": """
# RAGAS: Evaluating RAG Pipelines

RAGAS (Retrieval Augmented Generation Assessment) is an open-source framework for evaluating RAG pipelines without requiring human-labeled data (reference-free evaluation).

## Why Evaluation Matters

Building a RAG system is only half the challenge. Without systematic evaluation, it's impossible to know:
- Whether the system is hallucinating
- Whether it retrieves the right documents
- Whether answers actually address the question
- How different configurations compare

## Core RAGAS Metrics

### 1. Faithfulness (0–1)
**Definition**: Are all claims in the generated answer supported by the retrieved context?

**How it works**: RAGAS uses an LLM to decompose the answer into atomic claims, then verifies each claim against the context. Faithfulness = (claims supported by context) / (total claims).

**Interpretation**: Score of 0.9 means 90% of claims are grounded. Below 0.5 suggests significant hallucination.

### 2. Answer Relevancy (0–1)
**Definition**: How relevant and focused is the answer to the original question?

**How it works**: The LLM generates multiple questions from the answer, then measures cosine similarity between those generated questions and the original question. A focused, on-topic answer will yield questions close to the original.

**Interpretation**: Low scores indicate the answer drifted off-topic or was too vague.

### 3. Context Precision (0–1)
**Definition**: What fraction of the retrieved context is actually relevant?

**How it works**: Evaluates whether the retrieved chunks are signal or noise. High precision means the retriever returned focused, relevant chunks.

**Interpretation**: Low precision means irrelevant chunks are diluting the context window.

### 4. Context Recall (0–1)
**Definition**: Does the retrieved context contain all information needed to answer the question?

**How it works**: Compares the retrieved context against the ground truth answer to check if all necessary facts were retrieved.

**Interpretation**: Low recall means the retriever missed important information.

## Interpreting Combined Scores

| Scenario | Faithfulness | Ans. Rel. | Ctx. Prec. | Ctx. Rec. | Likely Issue |
|----------|:---:|:---:|:---:|:---:|:---:|
| Good overall | >0.8 | >0.8 | >0.8 | >0.8 | None |
| Hallucination | <0.5 | high | high | high | LLM adding facts |
| Poor retrieval | high | high | <0.5 | <0.5 | Wrong chunks |
| Off-topic answers | high | <0.5 | high | high | Prompt/LLM issue |
""",

    "05_optimization_techniques.txt": """
# RAG Optimization Techniques

Beyond basic RAG, several advanced techniques can significantly improve pipeline performance.

## Query Expansion

**Problem**: Users often ask short, ambiguous queries that don't match document vocabulary well.

**Solution**: Use an LLM to rewrite the query into a more specific, retrieval-friendly form before embedding it.

Example:
- Original: "ML training tips"
- Expanded: "What are best practices and techniques for training machine learning models efficiently, including learning rate scheduling and regularization?"

The expanded query better matches document language, improving retrieval recall.

## CrossEncoder Re-ranking

**Problem**: Bi-encoder embedding similarity is fast but approximate — it doesn't jointly reason about query and document together.

**Solution**: After initial retrieval (top-K candidates), apply a CrossEncoder model that reads query and document together and produces a more accurate relevance score. Re-rank candidates by this score.

CrossEncoders achieve higher precision than bi-encoders but are too slow to run on the full corpus — hence the two-stage approach.

Common models: cross-encoder/ms-marco-MiniLM-L-6-v2, cross-encoder/ms-marco-TinyBERT-L-2-v2

## Hypothetical Document Embeddings (HyDE)

Instead of embedding the raw query, use an LLM to generate a hypothetical answer, then embed that. The idea is that a well-formed answer is closer to real answer documents in embedding space than a short question.

## Parent-Child Chunking

**Problem**: Small chunks retrieve precisely but miss broader context.

**Solution**: Index small child chunks for retrieval, but return their parent chunk (larger context window) to the LLM. This gives precision in retrieval with breadth in generation.

## Ensemble Retrieval

Combine results from multiple retrievers:
- Dense retrieval (embeddings) + Sparse retrieval (BM25)
- Reciprocal Rank Fusion (RRF) to merge results

This captures both semantic similarity and keyword matching.

## Optimal Hyperparameters (Empirically Derived)

Based on extensive testing across document types:

| Parameter | Optimal Range | Notes |
|-----------|-------------|-------|
| chunk_size | 400–600 chars | Too small=fragmented; too large=diluted |
| chunk_overlap | 10–15% of chunk | 50–75 chars for size 512 |
| top_k | 3–5 | More causes noise; less misses context |
| retrieval_method | MMR for diversity | Similarity for precision |
| temperature | 0.0 | Determinism beats creativity for RAG |

## Hallucination Mitigation

1. Set temperature=0 for deterministic generation
2. Add explicit system prompt: "Only answer using the provided context"
3. Monitor faithfulness scores; flag responses below 0.5
4. Use chain-of-thought prompting to make reasoning traceable
5. Employ larger, better-instruction-following models (GPT-4 vs GPT-3.5)
"""
}


def create_sample_documents(output_dir: str = "data/raw"):
    """Create sample text documents for demo purposes."""
    os.makedirs(output_dir, exist_ok=True)

    for filename, content in SAMPLE_DOCS.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"✅ Created: {filepath}")

    print(f"\n📄 Created {len(SAMPLE_DOCS)} sample documents in {output_dir}/")
    print("   You can now run: python src/pipeline.py")


if __name__ == "__main__":
    create_sample_documents()
