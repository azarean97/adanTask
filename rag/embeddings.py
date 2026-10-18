"""
Standard Dense Embedding Generator using text-embedding-3-large.
Computes semantic embeddings through the OpenAI-compatible API endpoint
and calculates cosine similarity between queries and document vectors.
Saves embedding records to the outputs directory.
"""

import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import requests
from config import API_BASE_URL, API_KEY, EMBEDDING_MODEL, OUTPUTS_DIR

logger = logging.getLogger(__name__)

MANIFEST_FILE = OUTPUTS_DIR / "embeddings_manifest.json"

class EmbeddingGenerator:
    """Standard embedding generator using text-embedding-3-large."""
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.vectors: Dict[str, List[float]] = {}
        self._load_manifest()

    def _load_manifest(self):
        if MANIFEST_FILE.exists():
            try:
                with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                    self.vectors = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load embeddings manifest: {e}")
                self.vectors = {}

    def _save_manifest(self):
        try:
            with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
                json.dump(self.vectors, f)
        except Exception as e:
            logger.warning(f"Could not save embeddings manifest: {e}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings via API call."""
        results = []
        uncached_indices = []
        uncached_texts = []

        for idx, text in enumerate(texts):
            key = text.strip()
            if key in self.vectors:
                results.append(self.vectors[key])
            else:
                results.append(None)
                uncached_indices.append(idx)
                uncached_texts.append(key)

        if uncached_texts:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "input": uncached_texts if len(uncached_texts) > 1 else uncached_texts[0]
            }
            res = requests.post(
                f"{API_BASE_URL}/embeddings",
                headers=headers,
                json=payload,
                timeout=45
            )
            res.raise_for_status()
            data = res.json()["data"]

            for item_idx, item in zip(uncached_indices, data):
                vec = item["embedding"]
                text_key = texts[item_idx].strip()
                self.vectors[text_key] = vec
                results[item_idx] = vec

            self._save_manifest()

        return results

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        return self.embed_texts([query])[0]

# Singleton embedding instance
embedder = EmbeddingGenerator()

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Standard cosine similarity between two float vectors."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))
