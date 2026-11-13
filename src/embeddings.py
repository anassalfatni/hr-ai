"""Encodage local des CV et fiches de poste avec un modèle multilingue."""
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def _chunks(text: str, model: SentenceTransformer) -> list[str]:
    tokenizer = model.tokenizer
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    chunk_size = max(1, model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False))
    return [
        tokenizer.decode(token_ids[i:i + chunk_size], skip_special_tokens=True)
        for i in range(0, len(token_ids), chunk_size)
    ] or [text]


def encode_texts(texts: Sequence[str]) -> np.ndarray:
    model = _model()
    chunks_by_text = [_chunks(text, model) for text in texts]
    all_chunks = [chunk for chunks in chunks_by_text for chunk in chunks]
    chunk_vectors = model.encode(
        all_chunks,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(all_chunks) > 32,
    )

    vectors = []
    offset = 0
    for chunks in chunks_by_text:
        mean_vector = chunk_vectors[offset:offset + len(chunks)].mean(axis=0)
        norm = np.linalg.norm(mean_vector)
        vectors.append(mean_vector / norm if norm else mean_vector)
        offset += len(chunks)
    return np.asarray(vectors, dtype=np.float32)
