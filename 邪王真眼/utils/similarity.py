import json
import math

from typing import List, Dict, Tuple

from .tfidf_calculator import TFIDFCalculator


class TfidfDocumentSimilarity:
    def __init__(self, json_path: str, top_k_terms: int = 5):
        self.tf_idf_dict, self.vocab, self.corpus = self._load_data(json_path, top_k_terms)
        self.corpus_vectors = self._calculate_corpus_vectors()

    def _load_data(self, file_path: str, top_k: int) -> Tuple[Dict[int, Dict[str, float]], List[str], List[str]]:
        with open(file_path, encoding="utf8") as f:
            documents = json.load(f)

        corpus = []
        for doc in documents:
            title = doc["title"].replace("\n", " ")
            content = doc["content"].replace("\n", " ")
            corpus.append(f"{title}\n{content}")

        tokenized = TFIDFCalculator.tokenize_corpus(corpus)
        tfidf_dict = TFIDFCalculator.compute_tfidf(tokenized)

        vocab_set = set()
        for mapping in tfidf_dict.values():
            top_terms = sorted(mapping.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for w, _ in top_terms:
                vocab_set.add(w)
        vocab = list(vocab_set)

        return tfidf_dict, vocab, corpus

    def _doc_to_vec(self, passage: str) -> List[float]:
        tokens = TFIDFCalculator.tokenize_corpus([passage])[0]
        total = len(tokens) or 1
        return [tokens.count(term) / total for term in self.vocab]

    def _calculate_corpus_vectors(self) -> List[List[float]]:
        return [self._doc_to_vec(doc) for doc in self.corpus]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2 + 1e-7)

    def search_most_similar(self, query: str, top_n: int = 4) -> List[Tuple[int, float]]:
        qvec = self._doc_to_vec(query)
        scores = [(idx, self._cosine_similarity(qvec, dvec))
                  for idx, dvec in enumerate(self.corpus_vectors)]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]


if __name__ == "__main__":
    sim = TfidfDocumentSimilarity("./邪王真眼/datasets/news/news.json", top_k_terms=5)

    query = "魔兽争霸"
    results = sim.search_most_similar(query, top_n=4)
    for idx, score in results:
        title, content = sim.corpus[idx].split("\n", 1)
        print(f"=== 文档 #{idx} ({title}) 相似度: {score:.4f} ===")
        print(content[:200], "...\n")