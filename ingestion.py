[
  {
    "question": "What is Retrieval-Augmented Generation (RAG)?",
    "ground_truth": "RAG is a technique that combines a retrieval system with a language model. It first retrieves relevant documents from a corpus based on the user query, then uses those documents as context for the language model to generate grounded, factual answers. This reduces hallucinations and grounds responses in real documents.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What are the main components of a RAG pipeline?",
    "ground_truth": "The main components are: (1) a document corpus, (2) a text chunking and preprocessing module, (3) an embedding model to encode documents into vectors, (4) a vector database to store and retrieve embeddings, (5) a retriever to find relevant chunks given a query, and (6) an LLM to generate the final answer from the retrieved context.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What is the purpose of chunking in a RAG system?",
    "ground_truth": "Chunking splits large documents into smaller pieces that fit within the context window of embedding models and LLMs. Good chunking preserves semantic meaning while keeping each piece small enough for effective retrieval. Common strategies include fixed-size chunking, recursive character splitting, and sentence-based splitting.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What is the difference between similarity search and MMR retrieval?",
    "ground_truth": "Similarity search retrieves the top-k documents most similar to the query based on cosine similarity or dot product. MMR (Maximal Marginal Relevance) balances relevance and diversity — it selects documents that are relevant to the query but also different from each other, reducing redundancy in the retrieved context.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What metrics does RAGAS use to evaluate RAG systems?",
    "ground_truth": "RAGAS evaluates RAG systems using four key metrics: Faithfulness (does the answer stay true to the context?), Answer Relevancy (how relevant is the answer to the question?), Context Precision (is the retrieved context precise and free from irrelevant information?), and Context Recall (does the context contain all necessary information to answer the question?).",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What is faithfulness in the context of RAG evaluation?",
    "ground_truth": "Faithfulness measures whether the generated answer is factually consistent with the retrieved context. A high faithfulness score means the answer only uses information present in the context without hallucinating facts. It is computed by checking each factual claim in the answer against the provided documents.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What are the advantages of using ChromaDB as a vector store?",
    "ground_truth": "ChromaDB is an open-source vector database that supports persistent storage, metadata filtering, and easy integration with LangChain. It runs locally without external services, supports cosine similarity and MMR, and allows collections to be saved to disk and reloaded without re-indexing.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "How does query expansion improve RAG performance?",
    "ground_truth": "Query expansion rewrites the user's original query using an LLM to be more specific, detailed, or retrieval-friendly before embedding it. This improves retrieval quality by closing the vocabulary gap between how users phrase questions and how documents are written, leading to more relevant retrieved chunks.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What is CrossEncoder re-ranking and why is it useful?",
    "ground_truth": "CrossEncoder re-ranking applies a cross-attention model to score query-document pairs jointly. Unlike bi-encoder (embedding) models that encode query and document independently, CrossEncoders see both simultaneously and produce more accurate relevance scores. Re-ranking is applied after initial retrieval to reorder candidates by true relevance.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What embedding models are commonly used in RAG systems?",
    "ground_truth": "Common embedding models include OpenAI's text-embedding-ada-002 (commercial, high quality), and open-source alternatives like sentence-transformers/all-MiniLM-L6-v2 (fast, lightweight, 384 dimensions), BAAI/bge-large-en-v1.5 (high accuracy), and intfloat/e5-large (multilingual support).",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What is context precision in RAGAS evaluation?",
    "ground_truth": "Context Precision measures the signal-to-noise ratio in the retrieved context. It evaluates what proportion of the retrieved chunks are actually relevant to answering the question. A high context precision means fewer irrelevant chunks were retrieved, which helps the LLM focus on the right information.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What is context recall in RAGAS evaluation?",
    "ground_truth": "Context Recall measures whether the retrieved context contains all the information needed to answer the question, compared to the ground truth answer. A high recall means the retrieval system found all relevant documents. Low recall indicates important information was missed during retrieval.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "How does chunk overlap affect retrieval quality?",
    "ground_truth": "Chunk overlap ensures that information at the boundary between two chunks is not lost. Without overlap, a sentence split across chunks may be incomplete in either chunk, losing context. Too much overlap increases storage and redundancy. A typical value is 10-20% of the chunk size.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What is the role of temperature in LLM answer generation?",
    "ground_truth": "Temperature controls the randomness of the LLM's output. A temperature of 0 makes the model deterministic, always choosing the most likely token, which is ideal for factual RAG tasks. Higher temperatures introduce creativity and variation but may reduce faithfulness in a RAG context.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What is hallucination in the context of LLMs and RAG?",
    "ground_truth": "Hallucination occurs when an LLM generates information that is not supported by the provided context or factually incorrect. In RAG systems, faithfulness is used as a proxy metric to detect hallucinations. A faithfulness score below a threshold suggests the answer introduced facts not present in the retrieved documents.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "How does FAISS differ from ChromaDB as a vector store?",
    "ground_truth": "FAISS (Facebook AI Similarity Search) is a highly optimized library for dense vector similarity search, designed for speed at scale. ChromaDB is a complete vector database with metadata support, persistence, and filtering. FAISS is better for raw speed; ChromaDB is better for metadata-rich filtering and a managed experience.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What is the purpose of experiment tracking in ML projects?",
    "ground_truth": "Experiment tracking logs hyperparameters, metrics, and artifacts for each ML run so results can be compared, reproduced, and shared. Tools like MLflow track configuration (chunk size, top_k, model) alongside output metrics (faithfulness, latency) to identify the best-performing setup.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "Why is chunk size an important hyperparameter in RAG?",
    "ground_truth": "Chunk size determines how much text is included in each retrieved unit. Small chunks are precise but may miss context; large chunks include more context but may dilute relevance with irrelevant information. Optimal chunk size depends on document structure and the embedding model's maximum token limit.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "What is answer relevancy in RAGAS?",
    "ground_truth": "Answer Relevancy measures how directly the generated answer addresses the original question. It is computed by generating multiple questions from the answer using an LLM, then measuring the cosine similarity between these generated questions and the original question. A high score means the answer stays on topic.",
    "contexts": [],
    "answer": ""
  },
  {
    "question": "How can you reduce hallucinations in a RAG system?",
    "ground_truth": "Hallucinations can be reduced by: using temperature=0 for deterministic generation, adding explicit instructions to only use provided context, monitoring faithfulness scores and flagging low-scoring responses, improving retrieval quality to provide better context, and using larger or more capable LLMs with better instruction following.",
    "contexts": [],
    "answer": ""
  }
]
