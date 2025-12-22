# embeddings_utils.py
from __future__ import annotations

import math
from typing import List

from langchain_openai import OpenAIEmbeddings
from config import get_embedding_model


# Single embeddings client reused for all calls.
_embeddings = OpenAIEmbeddings(model=get_embedding_model())


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts into vector space."""
    return _embeddings.embed_documents(texts)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)
