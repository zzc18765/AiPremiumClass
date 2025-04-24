import math
import unicodedata

from collections import defaultdict
from typing import List, Dict, Optional

from .jieba_segmenter import JiebaSegmenter


class TFIDFCalculator:
    @staticmethod
    def tokenize_corpus(corpus: List[str]) -> List[List[str]]:
        def is_valid_token(tok: str) -> bool:
            for ch in tok:
                cat = unicodedata.category(ch)
                if not (cat.startswith('P')
                        or cat.startswith('Z')
                        or cat.startswith('C')):
                    return True
            return False

        tokenized: List[List[str]] = []
        for text in corpus:
            words = JiebaSegmenter.cut(text)
            
            filtered = [w for w in words if is_valid_token(w)]
            tokenized.append(filtered)

        return tokenized

    @staticmethod
    def compute_tf(tokenized_corpus: List[List[str]]) -> Dict[int, Dict[str, float]]:
        tf: Dict[int, Dict[str, float]] = {}
        for doc_idx, words in enumerate(tokenized_corpus):
            counts: Dict[str, float] = defaultdict(float)
            for w in words:
                counts[w] += 1.0

            total = sum(counts.values())
            for w in counts:
                counts[w] = counts[w] / total if total > 0 else 0.0
            tf[doc_idx] = dict(counts)
        return tf

    @staticmethod
    def compute_idf(tokenized_corpus: List[List[str]], smooth: bool = True) -> Dict[str, float]:
        N = len(tokenized_corpus)
        df: Dict[str, int] = defaultdict(int)
        
        for words in tokenized_corpus:
            for w in set(words):
                df[w] += 1

        idf: Dict[str, float] = {}
        for w, df_count in df.items():
            if smooth:
                idf[w] = math.log((N + 1) / (df_count + 1)) + 1
            else:
                idf[w] = math.log(N / (df_count + 1))
        return idf

    @staticmethod
    def compute_tfidf(
        tokenized_corpus: List[List[str]],
        tf: Optional[Dict[int, Dict[str, float]]] = None,
        idf: Optional[Dict[str, float]] = None
    ) -> Dict[int, Dict[str, float]]:
        if tf is None:
            tf = TFIDFCalculator.compute_tf(tokenized_corpus)
        if idf is None:
            idf = TFIDFCalculator.compute_idf(tokenized_corpus)

        tfidf: Dict[int, Dict[str, float]] = {}
        for doc_idx, tf_vals in tf.items():
            scores: Dict[str, float] = {}
            for w, tf_val in tf_vals.items():
                scores[w] = tf_val * idf.get(w, 0.0)
            tfidf[doc_idx] = scores

        return tfidf
