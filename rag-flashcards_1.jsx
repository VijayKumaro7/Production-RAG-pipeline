import { useState } from "react";

const flashcards = [
  // ─── RAG FUNDAMENTALS ───────────────────────────────────────
  {
    category: "RAG Fundamentals",
    color: "#7C3AED",
    q: "What is RAG (Retrieval-Augmented Generation)?",
    a: "A technique that retrieves relevant documents from an external knowledge base at inference time and conditions the LLM on that context to produce grounded, factual answers — reducing hallucinations and knowledge cutoff limitations.",
  },
  {
    category: "RAG Fundamentals",
    color: "#7C3AED",
    q: "What are the 2 phases of a RAG pipeline?",
    a: "1) Retrieval Phase: Embed the query → search vector store → fetch top-K chunks.\n2) Generation Phase: Feed retrieved chunks + question to LLM → generate grounded answer.",
  },
  {
    category: "RAG Fundamentals",
    color: "#7C3AED",
    q: "Why does RAG reduce hallucinations?",
    a: "The LLM is instructed to answer ONLY using the provided context. It no longer relies on parametric memory (weights) — every claim must be supported by retrieved documents, making answers traceable and verifiable.",
  },
  {
    category: "RAG Fundamentals",
    color: "#7C3AED",
    q: "What is the difference between parametric and non-parametric knowledge in LLMs?",
    a: "Parametric knowledge is baked into model weights during training (static, can hallucinate). Non-parametric knowledge is retrieved at runtime from external sources (dynamic, traceable). RAG adds non-parametric knowledge to any LLM.",
  },
  {
    category: "RAG Fundamentals",
    color: "#7C3AED",
    q: "Name 3 real-world use cases for RAG.",
    a: "1) Enterprise Q&A over internal documents\n2) Legal/medical document analysis\n3) Customer support automation\n4) Financial report analysis\n5) Research literature review assistant",
  },

  // ─── CHUNKING ───────────────────────────────────────────────
  {
    category: "Chunking",
    color: "#06B6D4",
    q: "What is chunking and why is it needed?",
    a: "Chunking splits large documents into smaller pieces that fit within embedding model token limits (typically 512 tokens). Smaller, focused chunks improve retrieval precision — the right chunk, not the whole document, is returned.",
  },
  {
    category: "Chunking",
    color: "#06B6D4",
    q: "What is Fixed-Size chunking?",
    a: "Splits text every N characters regardless of content boundaries. Simple and predictable, but may break sentences mid-thought. Best for uniform content like logs or code. Implemented via CharacterTextSplitter.",
  },
  {
    category: "Chunking",
    color: "#06B6D4",
    q: "What is Recursive Character chunking?",
    a: "Tries to split on natural boundaries in order: paragraph breaks → line breaks → sentences → words → characters. Preserves semantic units better than fixed splitting. LangChain's RecursiveCharacterTextSplitter is the standard choice.",
  },
  {
    category: "Chunking",
    color: "#06B6D4",
    q: "What is chunk overlap and why does it matter?",
    a: "Chunk overlap shares N characters between consecutive chunks. Without overlap, a sentence split across two chunks loses context at the boundary. Typical value: 10–20% of chunk_size (e.g., overlap=50 for size=512).",
  },
  {
    category: "Chunking",
    color: "#06B6D4",
    q: "What chunk_size gave the best RAGAS score in this project?",
    a: "chunk_size=512 with chunk_overlap=50 achieved the best balanced mean RAGAS score of 0.853 — outperforming both 256 (too fragmented) and 1024 (too diluted with irrelevant content).",
  },

  // ─── EMBEDDINGS ─────────────────────────────────────────────
  {
    category: "Embeddings",
    color: "#F59E0B",
    q: "What are vector embeddings?",
    a: "Dense numerical vectors that encode the semantic meaning of text. Semantically similar texts produce vectors close together in high-dimensional space. Embeddings enable similarity search — the core of retrieval in RAG.",
  },
  {
    category: "Embeddings",
    color: "#F59E0B",
    q: "What is all-MiniLM-L6-v2 and why use it?",
    a: "A free, lightweight HuggingFace sentence-transformer model (384 dimensions). Fast, runs on CPU, and achieves ~95% of OpenAI ada-002 quality on most tasks. Ideal for cost-free local RAG pipelines.",
  },
  {
    category: "Embeddings",
    color: "#F59E0B",
    q: "What is OpenAI text-embedding-ada-002?",
    a: "OpenAI's embedding model (1536 dimensions). High quality but paid per token. Useful when maximum retrieval accuracy is needed. In this project, it scored a mean RAGAS of 0.870 vs 0.853 for HuggingFace MiniLM.",
  },
  {
    category: "Embeddings",
    color: "#F59E0B",
    q: "What is cosine similarity and how is it used in retrieval?",
    a: "Cosine similarity measures the angle between two vectors (0=orthogonal, 1=identical direction). In RAG, the query vector is compared against all document vectors using cosine similarity to find the top-K most relevant chunks.",
  },
  {
    category: "Embeddings",
    color: "#F59E0B",
    q: "What does normalize_embeddings=True do in HuggingFace embeddings?",
    a: "It L2-normalizes each vector to unit length. This makes cosine similarity equivalent to dot product (faster), ensures consistent scale across different inputs, and improves retrieval quality for most models.",
  },

  // ─── VECTOR STORES ──────────────────────────────────────────
  {
    category: "Vector Stores",
    color: "#10B981",
    q: "What is ChromaDB?",
    a: "An open-source, Python-native vector database with persistent storage (SQLite-backed), metadata filtering, and LangChain integration. Runs locally, no external service needed. Best for development and medium-scale corpora.",
  },
  {
    category: "Vector Stores",
    color: "#10B981",
    q: "What is FAISS?",
    a: "Facebook AI Similarity Search — a high-performance library for dense vector search. Supports exact (IndexFlatL2) and approximate (IndexIVFFlat, HNSW) search. Can handle billions of vectors. No built-in metadata filtering.",
  },
  {
    category: "Vector Stores",
    color: "#10B981",
    q: "What is the difference between similarity search and MMR?",
    a: "Similarity search ranks purely by cosine distance (max relevance). MMR (Maximal Marginal Relevance) balances relevance + diversity — after each pick, it penalizes candidates similar to already-selected chunks, reducing redundancy.",
  },
  {
    category: "Vector Stores",
    color: "#10B981",
    q: "What is the MMR formula?",
    a: "MMR(d_i) = λ × sim(query, d_i) − (1−λ) × max(sim(d_i, d_j))\nλ=1 → pure relevance (similarity search)\nλ=0 → pure diversity\nλ=0.5 → balanced (default in this project)",
  },
  {
    category: "Vector Stores",
    color: "#10B981",
    q: "When would you choose FAISS over ChromaDB?",
    a: "FAISS when you need: maximum speed, very large corpora (100M+ vectors), GPU acceleration. ChromaDB when you need: metadata filtering, persistence without code, easier LangChain integration, development simplicity.",
  },

  // ─── RETRIEVAL ──────────────────────────────────────────────
  {
    category: "Retrieval",
    color: "#F43F5E",
    q: "What is top-K retrieval?",
    a: "The number of document chunks returned for a given query. top_k=4 (default) balances context richness vs noise. Low k → high precision, low recall. High k → high recall, lower precision. Optimal value found experimentally.",
  },
  {
    category: "Retrieval",
    color: "#F43F5E",
    q: "What is CrossEncoder re-ranking?",
    a: "A two-stage retrieval technique. First, fetch K*2 candidates with fast bi-encoder similarity. Then re-rank them using a CrossEncoder that reads query + document jointly (more accurate). Returns top-K after re-ranking.",
  },
  {
    category: "Retrieval",
    color: "#F43F5E",
    q: "Why are CrossEncoders more accurate than bi-encoders?",
    a: "Bi-encoders encode query and document independently (fast, scalable but approximate). CrossEncoders use cross-attention to see both simultaneously — capturing interaction between query tokens and document tokens for true relevance scoring.",
  },
  {
    category: "Retrieval",
    color: "#F43F5E",
    q: "What is query expansion?",
    a: "Using an LLM to rewrite the user's original query into a more specific, retrieval-friendly form before embedding. Bridges the vocabulary gap between how users ask questions and how documents are written. Improves recall for short/vague queries.",
  },
  {
    category: "Retrieval",
    color: "#F43F5E",
    q: "What is the vocabulary gap problem in RAG retrieval?",
    a: "Users ask questions differently than documents are written. E.g., user asks 'ML training tips' but doc says 'best practices for gradient descent optimization'. Query expansion rewrites the query to match document vocabulary, improving embedding similarity.",
  },

  // ─── LLM GENERATION ─────────────────────────────────────────
  {
    category: "LLM Generation",
    color: "#8B5CF6",
    q: "What temperature setting is best for RAG and why?",
    a: "temperature=0 is best for RAG. It makes output deterministic (always the highest-probability token). Higher temperatures introduce creative variation but reduce faithfulness — the LLM may add facts not in the context.",
  },
  {
    category: "LLM Generation",
    color: "#8B5CF6",
    q: "What is a RAG prompt template?",
    a: "A structured prompt with: (1) System instructions: 'Answer ONLY using the provided context', (2) Context: formatted retrieved chunks with source labels, (3) Human turn: the user's question. Forces grounded, traceable answers.",
  },
  {
    category: "LLM Generation",
    color: "#8B5CF6",
    q: "What is LangChain and what role does it play here?",
    a: "LangChain is an orchestration framework for LLM applications. In this project it provides: document loaders, text splitters, embedding wrappers, vector store integrations, prompt templates, and LLM chains — all with a unified API.",
  },
  {
    category: "LLM Generation",
    color: "#8B5CF6",
    q: "Name the 3 LLM providers supported in this project.",
    a: "1) OpenAI (GPT-3.5-turbo, GPT-4) — paid, best quality\n2) Ollama (Mistral, Llama) — free, runs locally\n3) Google Gemini (gemini-pro) — paid, strong alternative\nAll configured via config.yaml, no code changes needed.",
  },
  {
    category: "LLM Generation",
    color: "#8B5CF6",
    q: "What is the LangChain LCEL chain pattern used for generation?",
    a: "LCEL (LangChain Expression Language) chains components with | operator:\n`chain = prompt | llm | StrOutputParser()`\nThis pipes prompt → LLM → string parser. Clean, readable, and supports async/streaming.",
  },

  // ─── RAGAS EVALUATION ───────────────────────────────────────
  {
    category: "RAGAS Evaluation",
    color: "#06B6D4",
    q: "What is RAGAS?",
    a: "Retrieval Augmented Generation Assessment — an open-source framework for evaluating RAG pipelines. Uses an LLM as a judge to score responses. Reference-free (no human labels needed for most metrics). Outputs scores from 0 to 1.",
  },
  {
    category: "RAGAS Evaluation",
    color: "#06B6D4",
    q: "What is Faithfulness in RAGAS?",
    a: "Measures if all claims in the answer are supported by the retrieved context. LLM decomposes answer into atomic claims, then verifies each against context. Score = supported_claims / total_claims. Below 0.5 → potential hallucination.",
  },
  {
    category: "RAGAS Evaluation",
    color: "#06B6D4",
    q: "What is Answer Relevancy in RAGAS?",
    a: "Measures how directly the answer addresses the question. LLM generates multiple questions from the answer, then cosine similarity between those and the original question is measured. Low score → answer drifted off-topic or was vague.",
  },
  {
    category: "RAGAS Evaluation",
    color: "#06B6D4",
    q: "What is Context Precision in RAGAS?",
    a: "Measures the signal-to-noise ratio in retrieved context — what fraction of retrieved chunks are actually relevant. High precision = clean, focused retrieval. Low precision = irrelevant chunks diluting the context window.",
  },
  {
    category: "RAGAS Evaluation",
    color: "#06B6D4",
    q: "What is Context Recall in RAGAS?",
    a: "Measures whether the retrieved context contains ALL information needed to answer correctly, compared to the ground truth. Low recall = retriever missed important documents. Requires ground_truth answers in the evaluation dataset.",
  },
  {
    category: "RAGAS Evaluation",
    color: "#06B6D4",
    q: "What RAGAS scores did this project achieve?",
    a: "Best configuration (chunk_size=512, top_k=4, similarity):\n• Faithfulness: 0.91\n• Answer Relevancy: 0.88\n• Context Precision: 0.83\n• Context Recall: 0.79\n• Mean Score: 0.853 (+11% vs baseline)",
  },

  // ─── EXPERIMENTS ────────────────────────────────────────────
  {
    category: "Experiments",
    color: "#F97316",
    q: "What 5 hyperparameter experiments were run in this project?",
    a: "1) chunk_size: [256, 512, 1024]\n2) chunk_overlap: [0, 50, 100]\n3) top_k_retrieval: [2, 4, 6]\n4) retrieval_method: [similarity, MMR]\n5) embedding_model: [huggingface, openai]",
  },
  {
    category: "Experiments",
    color: "#F97316",
    q: "What is MLflow and how is it used here?",
    a: "MLflow is an open-source ML experiment tracking platform. In this project it logs: hyperparameters (chunk_size, top_k, method) and metrics (RAGAS scores) per run. Enables visual comparison of all experiments via the MLflow UI.",
  },
  {
    category: "Experiments",
    color: "#F97316",
    q: "What did the top_k experiment reveal?",
    a: "top_k=2: high precision (0.86) but low recall (0.68) — misses information.\ntop_k=4: best balance (precision 0.83, recall 0.79).\ntop_k=6: high recall (0.84) but lower precision (0.79) — more noise.",
  },
  {
    category: "Experiments",
    color: "#F97316",
    q: "What did the retrieval_method experiment reveal?",
    a: "Similarity: better precision (0.83), slightly lower recall (0.79). MMR: slightly lower precision (0.81) but better recall (0.85) due to diversity. MMR is better when documents have overlapping content. Similarity better for focused corpora.",
  },
  {
    category: "Experiments",
    color: "#F97316",
    q: "What is the purpose of a golden QA dataset in evaluation?",
    a: "A ground truth dataset of (question, answer) pairs created from your documents. Used to measure Context Recall (requires ground_truth) and validate evaluation quality. This project has 20 pairs covering core RAG concepts.",
  },

  // ─── ADVANCED FEATURES ──────────────────────────────────────
  {
    category: "Advanced Features",
    color: "#10B981",
    q: "What is hallucination detection in this project?",
    a: "After generation, the faithfulness score is computed by RAGAS. If it falls below the configured threshold (default: 0.5), the response is flagged as a potential hallucination. The Streamlit UI shows a warning banner for flagged responses.",
  },
  {
    category: "Advanced Features",
    color: "#10B981",
    q: "What is HyDE (Hypothetical Document Embeddings)?",
    a: "Instead of embedding the raw query, use an LLM to generate a hypothetical answer first, then embed that. The hypothesis is closer to real answer documents in embedding space than a short question — improves retrieval for complex queries.",
  },
  {
    category: "Advanced Features",
    color: "#10B981",
    q: "What is Parent-Child chunking?",
    a: "Index small child chunks for precise retrieval, but return their larger parent chunk as context to the LLM. Gets the best of both: precise retrieval (small chunks) + rich context (large chunks). Not yet implemented in this project.",
  },
  {
    category: "Advanced Features",
    color: "#10B981",
    q: "What is the cross-encoder model used in this project?",
    a: "cross-encoder/ms-marco-MiniLM-L-6-v2 — a lightweight CrossEncoder trained on MS MARCO passage ranking. Fast enough for real-time re-ranking. Scores query-document pairs with a single forward pass through BERT-like cross-attention.",
  },

  // ─── STREAMLIT & ARCHITECTURE ───────────────────────────────
  {
    category: "Architecture & Tools",
    color: "#F43F5E",
    q: "What are the 4 tabs in the Streamlit dashboard?",
    a: "1) 💬 Chat — conversational Q&A with source attribution\n2) 📊 Metrics — session stats and RAGAS scores\n3) 🧪 Experiments — Plotly bar + radar comparison charts\n4) 📄 Sources — inspect retrieved chunks with metadata",
  },
  {
    category: "Architecture & Tools",
    color: "#F43F5E",
    q: "What design pattern does RAGPipeline use?",
    a: "It uses a dataclass configuration (RAGConfig) + a class with an initialize() method. This separates config from logic, enables easy override for experiments, and supports caching in Streamlit via @st.cache_resource.",
  },
  {
    category: "Architecture & Tools",
    color: "#F43F5E",
    q: "Why is config.yaml used instead of hardcoding values?",
    a: "Centralizes all hyperparameters (chunk_size, top_k, model names, thresholds) in one place. Enables switching models or strategies without code changes. Supports overrides for experiment runner. Industry standard practice.",
  },
  {
    category: "Architecture & Tools",
    color: "#F43F5E",
    q: "What does python-dotenv do and why is it used?",
    a: "Loads environment variables from a .env file into os.environ. Keeps API keys (OPENAI_API_KEY, GOOGLE_API_KEY) out of source code and .gitignore-d from the repo. Never hardcode secrets — use dotenv.",
  },
  {
    category: "Architecture & Tools",
    color: "#F43F5E",
    q: "What files are excluded from the GitHub repo and why?",
    a: ".env (API keys), data/raw/ (user documents), data/chroma_db/ and faiss_index/ (generated indexes — large binary files), mlruns/ (experiment data), __pycache__/ (Python bytecode). All handled by .gitignore.",
  },
  {
    category: "Architecture & Tools",
    color: "#F43F5E",
    q: "What is the RAGResponse dataclass?",
    a: "A structured output object returned by pipeline.query() containing: question, expanded_question, answer, context, source_documents, latency_ms, metadata dict, and hallucination_flagged bool. Clean API contract between pipeline and UI.",
  },
];

const categoryColors = {
  "RAG Fundamentals": "#7C3AED",
  "Chunking": "#06B6D4",
  "Embeddings": "#F59E0B",
  "Vector Stores": "#10B981",
  "Retrieval": "#F43F5E",
  "LLM Generation": "#8B5CF6",
  "RAGAS Evaluation": "#06B6D4",
  "Experiments": "#F97316",
  "Advanced Features": "#10B981",
  "Architecture & Tools": "#F43F5E",
};

const categories = ["All", ...Object.keys(categoryColors)];

export default function FlashCards() {
  const [activeCategory, setActiveCategory] = useState("All");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [known, setKnown] = useState(new Set());
  const [reviewing, setReviewing] = useState(false);

  const filtered = reviewing
    ? flashcards.filter((_, i) => !known.has(i))
    : activeCategory === "All"
    ? flashcards
    : flashcards.filter((c) => c.category === activeCategory);

  const card = filtered[currentIndex];
  const progress = filtered.length > 0 ? ((currentIndex + 1) / filtered.length) * 100 : 0;
  const knownInFiltered = filtered.filter((_, i) =>
    known.has(flashcards.indexOf(filtered[i]))
  ).length;

  function next() {
    setFlipped(false);
    setTimeout(() => setCurrentIndex((i) => Math.min(i + 1, filtered.length - 1)), 120);
  }
  function prev() {
    setFlipped(false);
    setTimeout(() => setCurrentIndex((i) => Math.max(i - 1, 0)), 120);
  }
  function markKnown() {
    const globalIdx = flashcards.indexOf(card);
    setKnown((prev) => new Set([...prev, globalIdx]));
    next();
  }
  function changeCategory(cat) {
    setActiveCategory(cat);
    setCurrentIndex(0);
    setFlipped(false);
    setReviewing(false);
  }
  function reset() {
    setKnown(new Set());
    setCurrentIndex(0);
    setFlipped(false);
    setReviewing(false);
  }

  const totalKnown = known.size;
  const totalCards = flashcards.length;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0B0F1A",
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
      color: "#E2E8F0",
      padding: "0",
    }}>
      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #112240 0%, #0D1B2A 100%)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        padding: "20px 28px 16px",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: 22 }}>🔍</span>
              <span style={{ fontSize: 20, fontWeight: 800, letterSpacing: "-0.5px" }}>RAG Pipeline</span>
              <span style={{
                background: "linear-gradient(135deg, #7C3AED, #06B6D4)",
                color: "white", fontSize: 9, fontWeight: 700,
                padding: "2px 8px", borderRadius: 3, letterSpacing: 1,
                fontFamily: "monospace",
              }}>FLASHCARDS</span>
            </div>
            <div style={{ fontSize: 12, color: "#64748B", marginTop: 3, fontFamily: "monospace" }}>
              {totalCards} cards across {Object.keys(categoryColors).length} topics
            </div>
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <div style={{
              background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.3)",
              borderRadius: 8, padding: "6px 14px", fontSize: 12, color: "#10B981", fontFamily: "monospace",
            }}>
              ✓ {totalKnown} / {totalCards} known
            </div>
            <button onClick={reset} style={{
              background: "transparent", border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 8, padding: "6px 14px", fontSize: 12, color: "#94A3B8",
              cursor: "pointer",
            }}>↺ Reset</button>
            {totalKnown > 0 && (
              <button onClick={() => { setReviewing(!reviewing); setCurrentIndex(0); setFlipped(false); }} style={{
                background: reviewing ? "rgba(245,158,11,0.15)" : "transparent",
                border: `1px solid ${reviewing ? "#F59E0B" : "rgba(255,255,255,0.1)"}`,
                borderRadius: 8, padding: "6px 14px", fontSize: 12,
                color: reviewing ? "#F59E0B" : "#94A3B8", cursor: "pointer",
              }}>
                {reviewing ? "▶ All cards" : "📚 Review unknown"}
              </button>
            )}
          </div>
        </div>

        {/* Overall progress bar */}
        <div style={{ marginTop: 14 }}>
          <div style={{ height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2 }}>
            <div style={{
              height: "100%", borderRadius: 2,
              background: "linear-gradient(90deg, #7C3AED, #06B6D4)",
              width: `${(totalKnown / totalCards) * 100}%`,
              transition: "width 0.4s ease",
            }} />
          </div>
        </div>
      </div>

      {/* Category tabs */}
      <div style={{
        display: "flex", overflowX: "auto", gap: 6, padding: "14px 28px",
        background: "#0D1B2A", borderBottom: "1px solid rgba(255,255,255,0.05)",
        scrollbarWidth: "none",
      }}>
        {categories.map((cat) => {
          const isActive = activeCategory === cat && !reviewing;
          const col = cat === "All" ? "#7C3AED" : categoryColors[cat];
          const count = cat === "All" ? totalCards : flashcards.filter(c => c.category === cat).length;
          return (
            <button key={cat} onClick={() => changeCategory(cat)} style={{
              background: isActive ? `${col}22` : "transparent",
              border: `1px solid ${isActive ? col : "rgba(255,255,255,0.08)"}`,
              borderRadius: 20, padding: "5px 14px", whiteSpace: "nowrap",
              fontSize: 12, fontWeight: isActive ? 700 : 400,
              color: isActive ? col : "#64748B", cursor: "pointer",
              transition: "all 0.2s",
            }}>
              {cat} <span style={{ opacity: 0.6, fontSize: 10 }}>({count})</span>
            </button>
          );
        })}
      </div>

      {/* Main content */}
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "28px 20px" }}>
        {filtered.length === 0 ? (
          <div style={{ textAlign: "center", color: "#64748B", padding: "60px 0", fontSize: 14 }}>
            🎉 All cards marked as known! Press Reset to start again.
          </div>
        ) : (
          <>
            {/* Progress for current view */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <span style={{ fontFamily: "monospace", fontSize: 12, color: "#64748B" }}>
                {currentIndex + 1} / {filtered.length}
                {reviewing && <span style={{ color: "#F59E0B", marginLeft: 8 }}>• reviewing unknown</span>}
              </span>
              <div style={{
                background: `${card.color}22`,
                border: `1px solid ${card.color}55`,
                borderRadius: 20, padding: "3px 12px",
                fontSize: 11, color: card.color, fontFamily: "monospace",
              }}>
                {card.category}
              </div>
            </div>

            {/* Progress bar */}
            <div style={{ height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 2, marginBottom: 24 }}>
              <div style={{
                height: "100%", borderRadius: 2,
                background: card.color, width: `${progress}%`,
                transition: "width 0.3s ease",
              }} />
            </div>

            {/* Card */}
            <div
              onClick={() => setFlipped(!flipped)}
              style={{
                background: flipped ? `${card.color}12` : "#112240",
                border: `1px solid ${flipped ? card.color + "55" : "rgba(255,255,255,0.07)"}`,
                borderRadius: 16, padding: "36px 40px",
                minHeight: 260, cursor: "pointer",
                transition: "all 0.25s ease",
                position: "relative", overflow: "hidden",
                boxShadow: flipped ? `0 0 40px ${card.color}22` : "0 4px 24px rgba(0,0,0,0.3)",
              }}
            >
              {/* Top accent */}
              <div style={{
                position: "absolute", top: 0, left: 0, right: 0, height: 3,
                background: card.color, borderRadius: "16px 16px 0 0",
              }} />

              <div style={{
                position: "absolute", top: 14, right: 16,
                fontFamily: "monospace", fontSize: 10, color: flipped ? card.color : "#334155",
                background: flipped ? `${card.color}22` : "rgba(255,255,255,0.04)",
                padding: "3px 10px", borderRadius: 4,
              }}>
                {flipped ? "ANSWER" : "QUESTION — click to flip"}
              </div>

              {!flipped ? (
                <div>
                  <div style={{ fontSize: 13, color: "#64748B", fontFamily: "monospace", marginBottom: 16 }}>
                    Q{flashcards.indexOf(card) + 1}
                  </div>
                  <div style={{ fontSize: 19, fontWeight: 700, lineHeight: 1.5, color: "#F1F5F9" }}>
                    {card.q}
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 13, color: card.color, fontFamily: "monospace", marginBottom: 16 }}>
                    ANSWER
                  </div>
                  <div style={{
                    fontSize: 15, lineHeight: 1.75, color: "#CBD5E1",
                    whiteSpace: "pre-line",
                  }}>
                    {card.a}
                  </div>
                </div>
              )}
            </div>

            {/* Controls */}
            <div style={{ display: "flex", gap: 10, marginTop: 20, justifyContent: "center" }}>
              <button onClick={prev} disabled={currentIndex === 0} style={{
                background: "transparent", border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 10, padding: "10px 22px", fontSize: 14, color: "#94A3B8",
                cursor: currentIndex === 0 ? "not-allowed" : "pointer",
                opacity: currentIndex === 0 ? 0.4 : 1,
              }}>← Prev</button>

              <button onClick={markKnown} style={{
                background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.4)",
                borderRadius: 10, padding: "10px 26px", fontSize: 14, color: "#10B981",
                cursor: "pointer", fontWeight: 600,
              }}>✓ Got it</button>

              <button onClick={next} disabled={currentIndex === filtered.length - 1} style={{
                background: `${card.color}22`, border: `1px solid ${card.color}55`,
                borderRadius: 10, padding: "10px 22px", fontSize: 14, color: card.color,
                cursor: currentIndex === filtered.length - 1 ? "not-allowed" : "pointer",
                opacity: currentIndex === filtered.length - 1 ? 0.4 : 1, fontWeight: 600,
              }}>Next →</button>
            </div>

            <div style={{ textAlign: "center", marginTop: 14, fontSize: 11, color: "#334155", fontFamily: "monospace" }}>
              tap card to flip · use buttons to navigate · ✓ Got it to track progress
            </div>
          </>
        )}

        {/* All categories summary */}
        <div style={{ marginTop: 48, display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 10 }}>
          {Object.entries(categoryColors).map(([cat, col]) => {
            const total = flashcards.filter(c => c.category === cat).length;
            const knownCount = flashcards.filter((c, i) => c.category === cat && known.has(i)).length;
            return (
              <div key={cat} onClick={() => changeCategory(cat)} style={{
                background: "#0D1B2A", border: `1px solid ${activeCategory === cat ? col : "rgba(255,255,255,0.06)"}`,
                borderRadius: 10, padding: "12px 16px", cursor: "pointer",
                transition: "all 0.2s",
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: col }}>{cat}</span>
                  <span style={{ fontFamily: "monospace", fontSize: 11, color: "#64748B" }}>{knownCount}/{total}</span>
                </div>
                <div style={{ marginTop: 8, height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 2 }}>
                  <div style={{
                    height: "100%", borderRadius: 2, background: col,
                    width: `${total > 0 ? (knownCount / total) * 100 : 0}%`,
                    transition: "width 0.3s",
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
