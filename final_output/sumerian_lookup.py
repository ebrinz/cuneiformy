"""
Sumerian Semantic Lookup: Find Sumerian words by English meaning.

Uses pickle for loading locally-generated vocab files (matching heiroglyphy pattern).
"""
import pickle
import numpy as np


class SumerianLookup:
    def __init__(self, vectors_path, vocab_path, glove_vectors, glove_vocab):
        data = np.load(vectors_path, allow_pickle=True)
        self.vectors = data["vectors"].astype(np.float32)
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors_norm = self.vectors / norms

        self.glove_vectors = glove_vectors
        self.glove_vocab = glove_vocab
        self.glove_word_to_idx = {w: i for i, w in enumerate(glove_vocab)}

        g_norms = np.linalg.norm(self.glove_vectors, axis=1, keepdims=True)
        g_norms[g_norms == 0] = 1
        self.glove_norm = self.glove_vectors / g_norms

    def _get_english_vector(self, word):
        idx = self.glove_word_to_idx.get(word.lower())
        if idx is None:
            return None
        return self.glove_norm[idx]

    def find(self, english_word, top_k=10):
        vec = self._get_english_vector(english_word)
        if vec is None:
            return []
        sims = self.vectors_norm @ vec
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.vocab[i], float(sims[i])) for i in top_indices]

    def find_analogy(self, a, b, c, top_k=10):
        va = self._get_english_vector(a)
        vb = self._get_english_vector(b)
        vc = self._get_english_vector(c)
        if any(v is None for v in [va, vb, vc]):
            return []
        target = vc - va + vb
        target = target / (np.linalg.norm(target) + 1e-10)
        sims = self.vectors_norm @ target
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.vocab[i], float(sims[i])) for i in top_indices]

    def find_blend(self, weights, top_k=10):
        target = np.zeros(self.vectors.shape[1], dtype=np.float32)
        for word, weight in weights.items():
            vec = self._get_english_vector(word)
            if vec is not None:
                target += weight * vec
        norm = np.linalg.norm(target)
        if norm == 0:
            return []
        target = target / norm
        sims = self.vectors_norm @ target
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.vocab[i], float(sims[i])) for i in top_indices]
