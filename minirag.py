
from functools import lru_cache

from config import RAG_MIN_SIMILARITY

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

KNOWLEDGE_BASE = [
    "Температура (temperature) - это гиперпараметр LLM. Высокая температура (0.7+) делает ответы креативными, низкая (0.1) - точными.",
    "RAG (Retrieval-Augmented Generation) - это метод, где модель сначала ищет факты в базе, а потом генерирует ответ.",
    "Сбер - это крупнейший банк в России, который активно развивает AI и GigaChat.",
    "Python - это язык программирования, который любит утка."
]

@lru_cache(maxsize=1)
def _load_index():
    """Load the encoder and document vectors on the first search only."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    document_embeddings = model.encode(
        KNOWLEDGE_BASE,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np, model, document_embeddings


def rag_search(query: str) -> str | None:
    """Return the best matching item, or None when it is below the relevance threshold."""
    if not query.strip():
        return None

    np, model, document_embeddings = _load_index()
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    similarities = document_embeddings @ query_embedding
    best_match_idx = int(np.argmax(similarities))
    best_similarity = float(similarities[best_match_idx])

    if best_similarity < RAG_MIN_SIMILARITY:
        return None

    return KNOWLEDGE_BASE[best_match_idx]