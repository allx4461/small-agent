import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Загружаем маленькую модель для эмбеддингов (работает локально, бесплатно)
# Она превращает текст в вектор из 384 чисел
model = SentenceTransformer('all-MiniLM-L6-v2')

KNOWLEDGE_BASE = [
    "Температура (temperature) - это гиперпараметр LLM. Высокая температура (0.7+) делает ответы креативными, низкая (0.1) - точными.",
    "RAG (Retrieval-Augmented Generation) - это метод, где модель сначала ищет факты в базе, а потом генерирует ответ.",
    "Сбер - это крупнейший банк в России, который активно развивает AI и GigaChat.",
    "Python - это язык программирования, который любит утка."
]

# 3. Предварительно считаем векторы для всех документов в базе (делаем это 1 раз при запуске)
# Сохраняем их в матрицу для быстрых вычислений
db_embeddings = model.encode(KNOWLEDGE_BASE)

def rag_search(query: str) -> str:
    """НRAG поиск на косинусном сходстве."""
    # Превращаем запрос в вектор
    query_embedding = model.encode(query)
    
    # Считаем косинусное сходство между запросом и каждым документом в базе
    # Формула: (A * B) / (|A| * |B|)
    similarities = np.dot(db_embeddings, query_embedding) / (
        np.linalg.norm(db_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Находим индекс документа с максимальным сходством
    best_match_idx = np.argmax(similarities)
    
    # Возвращаем текст этого документа
    return KNOWLEDGE_BASE[best_match_idx]

# Тест
# print(rag_search("как сделать ответы креативными?")) 
# Вернет текст про температуру, потому что векторы будут близки по смыслу!