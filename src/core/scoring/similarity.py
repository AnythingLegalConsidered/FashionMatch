"""
Cosine similarity computation functions.

Provides efficient similarity computation between embeddings using numpy.

Example:
    >>> from src.core.scoring.similarity import cosine_similarity
    >>> sim = cosine_similarity(embedding_a, embedding_b)
    >>> print(f"Similarity: {sim:.4f}")
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

# Type alias for vectors
VectorLike = Union[np.ndarray, List[float]]


def cosine_similarity(
    vec_a: VectorLike,
    vec_b: VectorLike,
) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity measures the angle between vectors, ranging from:
    - 1.0: Identical direction (most similar)
    - 0.0: Orthogonal (no similarity)
    - -1.0: Opposite direction (least similar)
    
    For normalized vectors (L2 norm = 1), this is equivalent to dot product.
    
    Args:
        vec_a: First vector (numpy array or list).
        vec_b: Second vector (numpy array or list).
        
    Returns:
        Cosine similarity score in range [-1, 1].
        
    Raises:
        ValueError: If vectors have different dimensions.
        
    Example:
        >>> a = np.array([1.0, 0.0, 0.0])
        >>> b = np.array([1.0, 0.0, 0.0])
        >>> cosine_similarity(a, b)
        1.0
        
        >>> c = np.array([0.0, 1.0, 0.0])
        >>> cosine_similarity(a, c)
        0.0
    """
    # Convert to numpy arrays if needed
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    
    # Validate dimensions
    if a.shape != b.shape:
        raise ValueError(
            f"Vector dimensions must match: {a.shape} vs {b.shape}"
        )
    
    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Compute cosine similarity
    similarity = np.dot(a, b) / (norm_a * norm_b)
    
    # Clip to [-1, 1] to handle floating point errors
    return float(np.clip(similarity, -1.0, 1.0))


def cosine_similarity_normalized(
    vec_a: VectorLike,
    vec_b: VectorLike,
) -> float:
    """
    Compute cosine similarity for already L2-normalized vectors.
    
    Faster than cosine_similarity() when vectors are pre-normalized
    (like our encoder outputs).
    
    Args:
        vec_a: First L2-normalized vector.
        vec_b: Second L2-normalized vector.
        
    Returns:
        Cosine similarity score in range [-1, 1].
        
    Example:
        >>> # Our encoders output normalized vectors
        >>> clip_emb_1 = encoder.encode(image_1)  # Already normalized
        >>> clip_emb_2 = encoder.encode(image_2)
        >>> sim = cosine_similarity_normalized(clip_emb_1, clip_emb_2)
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    
    # For normalized vectors, cosine similarity = dot product
    similarity = np.dot(a, b)
    
    return float(np.clip(similarity, -1.0, 1.0))


def batch_cosine_similarity(
    query: VectorLike,
    candidates: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple candidates.
    
    Efficient batch computation using matrix operations.
    
    Args:
        query: Query vector of shape (dim,).
        candidates: Matrix of candidate vectors, shape (n, dim).
        
    Returns:
        Array of similarity scores, shape (n,).
        
    Example:
        >>> query = encoder.encode(reference_image)
        >>> candidates = encoder.encode_batch(vinted_images)
        >>> similarities = batch_cosine_similarity(query, candidates)
        >>> top_indices = np.argsort(similarities)[::-1][:10]
    """
    q = np.asarray(query, dtype=np.float32)
    c = np.asarray(candidates, dtype=np.float32)
    
    # Normalize query
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    
    # Normalize candidates (row-wise)
    c_norms = np.linalg.norm(c, axis=1, keepdims=True) + 1e-8
    c_normalized = c / c_norms
    
    # Compute all similarities at once
    similarities = np.dot(c_normalized, q_norm)
    
    return similarities


def batch_cosine_similarity_normalized(
    query: VectorLike,
    candidates: np.ndarray,
) -> np.ndarray:
    """
    Batch cosine similarity for pre-normalized vectors.
    
    Fastest option when all vectors are already L2-normalized.
    
    Args:
        query: L2-normalized query vector of shape (dim,).
        candidates: L2-normalized matrix of shape (n, dim).
        
    Returns:
        Array of similarity scores, shape (n,).
    """
    q = np.asarray(query, dtype=np.float32)
    c = np.asarray(candidates, dtype=np.float32)
    
    # For normalized vectors, just compute dot products
    return np.dot(c, q)


def euclidean_distance(
    vec_a: VectorLike,
    vec_b: VectorLike,
) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        vec_a: First vector.
        vec_b: Second vector.
        
    Returns:
        Euclidean distance (0 = identical, higher = more different).
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    
    return float(np.linalg.norm(a - b))


def similarity_to_distance(similarity: float) -> float:
    """
    Convert cosine similarity to distance-like metric.
    
    Useful for algorithms that expect distances instead of similarities.
    
    Args:
        similarity: Cosine similarity in [-1, 1].
        
    Returns:
        Distance in [0, 2] where 0 = identical.
    """
    return 1.0 - similarity


def distance_to_similarity(distance: float) -> float:
    """
    Convert distance to similarity-like metric.
    
    Args:
        distance: Distance value (0 = identical).
        
    Returns:
        Similarity where higher = more similar.
    """
    return 1.0 - distance
