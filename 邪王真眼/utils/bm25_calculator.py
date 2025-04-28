import math
import unicodedata

from collections import Counter
from typing import Dict, List, Tuple, Optional

from utils.jieba_segmenter import JiebaSegmenter


class BM25Calculator:
    @staticmethod
    def tokenize_documents(
        documents: Dict[str, List[str]],
        stopwords: Optional[List[str]] = []
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        def is_valid_token(tok: str) -> bool:
            for ch in tok:
                if ch in stopwords:
                    return False
                cat = unicodedata.category(ch)
                if not (cat.startswith('P')
                        or cat.startswith('Z')
                        or cat.startswith('C')):
                    return True
            return False

        doc_terms = {}
        all_terms = []
        
        for doc_name, doc_contents in documents.items():
            terms_for_doc = []
            for content in doc_contents:
                words = [word for word in JiebaSegmenter.cut(content) if is_valid_token(word)]
                terms_for_doc.extend(words)
                all_terms.extend(words)
            doc_terms[doc_name] = terms_for_doc
            
        return doc_terms, all_terms

    @staticmethod
    def compute_document_stats(
        doc_terms: Dict[str, List[str]]
    ) -> Tuple[Dict[str, int], float]:
        doc_lengths = {}
        total_length = 0
        
        for doc_name, terms in doc_terms.items():
            length = len(terms)
            doc_lengths[doc_name] = length
            total_length += length
            
        avgdl = total_length / len(doc_terms) if doc_terms else 0.0
        return doc_lengths, avgdl

    @staticmethod
    def compute_term_frequencies(
        doc_terms: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, int]]:
        doc_term_freq = {}
        
        for doc_name, terms in doc_terms.items():
            doc_term_freq[doc_name] = Counter(terms)
            
        return doc_term_freq

    @staticmethod
    def compute_idf(
        doc_term_freq: Dict[str, Dict[str, int]], 
        N: int
    ) -> Dict[str, float]:
        idf = {}
        unique_terms = set(term for term_freq in doc_term_freq.values() for term in term_freq)
        
        for term in unique_terms:
            doc_count = sum(1 for freq in doc_term_freq.values() if term in freq)
            idf[term] = math.log((N - doc_count + 0.5) / (doc_count + 0.5))
            
        return idf

    @staticmethod
    def compute_bm25(
        documents: Dict[str, List[str]],
        stopwords: Optional[List[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        doc_term_freq: Optional[Dict[str, Dict[str, int]]] = None,
        doc_lengths: Optional[Dict[str, int]] = None,
        avgdl: Optional[float] = None,
        idf: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
        doc_terms, all_terms = BM25Calculator.tokenize_documents(documents, stopwords)
        
        if doc_lengths is None or avgdl is None:
            doc_lengths, avgdl = BM25Calculator.compute_document_stats(doc_terms)
            
        if doc_term_freq is None:
            doc_term_freq = BM25Calculator.compute_term_frequencies(doc_terms)
            
        if idf is None:
            idf = BM25Calculator.compute_idf(doc_term_freq, len(documents))
            
        bm25_scores = {}
        
        for doc_name, term_freq in doc_term_freq.items():
            bm25_scores[doc_name] = {}
            doc_len = doc_lengths[doc_name]
            
            for term, tf in term_freq.items():
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                bm25_scores[doc_name][term] = idf.get(term, 0.0) * (numerator / denominator)
                
        unique_terms = list(set(all_terms))
        return bm25_scores, unique_terms
