import re
import json

from typing import List, Dict, Optional, Tuple

from .jieba_segmenter import JiebaSegmenter
from .tfidf_calculator import TFIDFCalculator


class TfidfSummarizer:
    def load_data(self, file_path: str) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
        with open(file_path, encoding="utf8") as f:
            documents = json.load(f)

        corpus: List[str] = []
        for doc in documents:
            title = doc["title"].replace("\n", " ")
            content = doc["content"].replace("\n", " ")
            corpus.append(f"{title}\n{content}")

        tf_idf_dict, _ = TFIDFCalculator.compute_tfidf(corpus)

        return tf_idf_dict, corpus

    @staticmethod
    def generate_document_abstract(
        document_tf_idf: Dict[str, float],
        content: str,
        top: int = 3
    ) -> Optional[str]:
        sentences = [s.strip() for s in re.split(r"[？！？。]", content) if s.strip()]
        if len(sentences) <= 5:
            return None

        scored: List[tuple] = []
        for idx, sent in enumerate(sentences):
            words = JiebaSegmenter.cut(sent)
            score = sum(document_tf_idf.get(w, 0.0) for w in words) / (len(words) + 1)
            scored.append((score, idx))

        top_idxs = sorted(idx for _, idx in sorted(scored, reverse=True)[:top])
        abstract = "。".join(sentences[i] for i in top_idxs)
        return abstract

    def summarize(
        self,
        json_path: str,
        top: int = 3
    ) -> List[Dict[str, str]]:
        tf_idf_dict, corpus = self.load_data(json_path)
        results: List[Dict[str, str]] = []

        for doc_idx, tfidf in tf_idf_dict.items():
            title, content = corpus[doc_idx].split("\n", 1)
            abstract = self.generate_document_abstract(tfidf, content, top=top)
            if abstract:
                results.append({
                    "标题": title,
                    "正文": content,
                    "摘要": abstract
                })

        return results


if __name__ == "__main__":
    summarizer = TfidfSummarizer()
    summaries = summarizer.summarize("./邪王真眼/datasets/news/news.json", top=3)

    with open("./邪王真眼/datasets/news/abstract.json", "w", encoding="utf8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"已生成 {len(summaries)} 篇文章摘要，输出到 abstract.json")
