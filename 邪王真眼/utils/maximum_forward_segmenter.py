from typing import Set, Dict, List


class MaximumForwardSegmenter:
    @staticmethod
    def compute_max_len(word_set: Set[str]) -> int:
        return max((len(w) for w in word_set), default=1)

    @staticmethod
    def max_forward_cut1(word_set: Set[str], text: str, max_len: int) -> List[str]:
        if max_len < 1:
            raise ValueError("max_len must be at least 1")

        segments: List[str] = []
        idx = 0
        n = len(text)
        while idx < n:
            length = min(max_len, n - idx)
            candidate = text[idx:idx + length]
            while candidate not in word_set and len(candidate) > 1:
                candidate = candidate[:-1]
            segments.append(candidate)
            idx += len(candidate)
        return segments

    @staticmethod
    def max_forward_cut2(word_set: Set[str], text: str) -> List[str]:
        prefix_dict: Dict[str, int] = {}
        for w in word_set:
            for i in range(1, len(w)):
                prefix = w[:i]
                if prefix not in prefix_dict:
                    prefix_dict[prefix] = 0
            prefix_dict[w] = 1

        segments: List[str] = []
        start = 0
        n = len(text)

        while start < n:
            end = start + 1
            last_match = text[start:start+1]
            while end <= n and text[start:end] in prefix_dict:
                if prefix_dict[text[start:end]] == 1:
                    last_match = text[start:end]
                end += 1

            segments.append(last_match)
            start += len(last_match)

        return segments
    
    @staticmethod
    def max_backward_cut1(word_set: Set[str], text: str, max_len: int) -> List[str]:
        if max_len < 1:
            raise ValueError("max_len must be at least 1")
        segments: List[str] = []
        idx = len(text)
        while idx > 0:
            length = min(max_len, idx)
            candidate = text[idx - length:idx]
            while candidate not in word_set and len(candidate) > 1:
                candidate = candidate[1:]
            segments.insert(0, candidate)
            idx -= len(candidate)
        return segments

    @staticmethod
    def max_backward_cut2(word_set: Set[str], text: str) -> List[str]:
        prefix_rev: Dict[str, int] = {}
        for w in word_set:
            rev_w = w[::-1]
            for i in range(1, len(rev_w)):
                pre = rev_w[:i]
                if pre not in prefix_rev:
                    prefix_rev[pre] = 0
            prefix_rev[rev_w] = 1

        rev_text = text[::-1]
        segments_rev: List[str] = []
        start, n = 0, len(rev_text)
        while start < n:
            end = start + 1
            last_match = rev_text[start:start+1]
            while end <= n and rev_text[start:end] in prefix_rev:
                if prefix_rev[rev_text[start:end]] == 1:
                    last_match = rev_text[start:end]
                end += 1
            segments_rev.append(last_match)
            start += len(last_match)

        return [seg[::-1] for seg in reversed(segments_rev)]


if __name__ == "__main__":
    with open('./邪王真眼/datasets/cut_words/dict.txt', encoding='utf8') as f:
        word_set = {line.strip().split()[0] for line in f}
    max_len = MaximumForwardSegmenter.compute_max_len(word_set)
    text = "测试字符串"
    print('Forward1:', MaximumForwardSegmenter.max_forward_cut1(word_set, text, max_len))
    print('Forward2:', MaximumForwardSegmenter.max_forward_cut2(word_set, text))
    print('Backward1:', MaximumForwardSegmenter.max_backward_cut1(word_set, text, max_len))
    print('Backward2:', MaximumForwardSegmenter.max_backward_cut2(word_set, text))