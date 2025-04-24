import json

from typing import List, Tuple, Dict

from .tfidf_calculator import TFIDFCalculator


class TfidfSearchEngine:
    def __init__(self, json_path: str, top_n: int = 3):
        self.top_n = top_n
        self.tf_idf_dict, self.corpus = self._load_data(json_path)

    def _load_data(self, file_path: str) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
        with open(file_path, encoding="utf8") as f:
            documents = json.load(f)
        corpus = [doc["title"] + "\n" + doc["content"] for doc in documents]
        tokenized = TFIDFCalculator.tokenize_corpus(corpus)
        tf = TFIDFCalculator.compute_tf(tokenized)
        idf = TFIDFCalculator.compute_idf(tokenized)
        tfidf = TFIDFCalculator.compute_tfidf(tokenized, tf, idf)
        return tfidf, corpus

    def search(self, query: str) -> List[Tuple[int, float]]:
        tokens = TFIDFCalculator.tokenize_corpus([query])[0]
        scores = [
            (doc_id, sum(self.tf_idf_dict[doc_id].get(w, 0.0) for w in tokens))
            for doc_id in self.tf_idf_dict
        ]
        results = sorted(scores, key=lambda x: x[1], reverse=True)[: self.top_n]
        for doc_id, _ in results:
            print(self.corpus[doc_id])
            print("--------------")
        return results


if __name__ == "__main__":
    engine = TfidfSearchEngine("./邪王真眼/datasets/news/news.json", top_n=3)
    while True:
        q = input("请输入您要搜索的内容: ")
        engine.search(q)
