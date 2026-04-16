"""
Sentence Transformers embedding service for the study pipeline.

Provides consistent embedding generation using sentence-transformers
for text chunks, queries, concepts, and answers throughout the application.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

# Default model - high performance with good speed/accuracy tradeoff
# all-MiniLM-L6-v2: 384 dimensions, fast, good general purpose
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Singleton model instance
_model_instance = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Get or create the singleton embedding model instance.
    Lazy loads the model on first access.
    """
    global _model_instance
    if _model_instance is None:
        print(f"Loading embedding model: {model_name}")
        _model_instance = SentenceTransformer(model_name)
        print(f"Embedding model loaded successfully. Dimensions: {_model_instance.get_sentence_embedding_dimension()}")
    return _model_instance


def generate_embedding(text: Union[str, List[str]], normalize: bool = True) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings for single text or batch of texts.
    
    Args:
        text: Single string or list of strings to embed
        normalize: Whether to L2 normalize embeddings (recommended for cosine similarity)
    
    Returns:
        Embedding vector(s) as lists of floats
    """
    model = get_embedding_model()
    
    if isinstance(text, str):
        # Single text
        embedding = model.encode(text, normalize_embeddings=normalize)
        return embedding.tolist()
    else:
        # Batch processing
        embeddings = model.encode(text, normalize_embeddings=normalize, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]


def generate_chunk_embeddings(chunks: List[dict]) -> List[List[float]]:
    """
    Generate embeddings for document chunks.
    
    Args:
        chunks: List of chunk dictionaries with 'text' field
    
    Returns:
        List of embedding vectors
    """
    texts = [chunk['text'] for chunk in chunks]
    return generate_embedding(texts)


def get_embedding_dimension() -> int:
    """Return the dimension size of the current embedding model."""
    model = get_embedding_model()
    return model.get_sentence_embedding_dimension()


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embedding vectors.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Similarity score between 0 and 1 (higher = more similar)
    """
    a = np.array(embedding1)
    b = np.array(embedding2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def batch_cosine_similarity(query_embedding: List[float], embeddings: List[List[float]]) -> List[float]:
    """
    Calculate cosine similarity between a query and multiple embeddings efficiently.
    
    Args:
        query_embedding: Query embedding vector
        embeddings: List of embedding vectors to compare against
    
    Returns:
        List of similarity scores in the same order as input embeddings
    """
    query = np.array(query_embedding)
    targets = np.array(embeddings)
    
    dot_products = np.dot(targets, query)
    query_norm = np.linalg.norm(query)
    target_norms = np.linalg.norm(targets, axis=1)
    
    similarities = dot_products / (query_norm * target_norms)
    return similarities.tolist()