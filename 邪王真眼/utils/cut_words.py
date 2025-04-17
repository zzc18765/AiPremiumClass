import jieba
from typing import List, Tuple, Dict


def texts_to_index_sequences(
    texts: List[str]
) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
    token_lists, vocab = [], set()
    for text in texts:
        tokens = list(jieba.cut(str(text)))
        token_lists.append(tokens)
        vocab.update(tokens)

    vocab_to_index = {w: i + 2 for i, w in enumerate(vocab)}
    vocab_to_index["UNK"] = 0
    vocab_to_index["PAD"] = 1
    index_to_vocab = {i: w for w, i in vocab_to_index.items()}

    max_len = max(len(t) for t in token_lists)
    sequences = []
    for tokens in token_lists:
        idx_seq = [vocab_to_index.get(tok, 0) for tok in tokens]
        idx_seq.extend([1] * (max_len - len(idx_seq)))
        sequences.append(idx_seq)

    return sequences, vocab_to_index, index_to_vocab, max_len


def build_eval_sequences(
    texts: List[str],
    vocab_to_index: Dict[str, int],
    text_len: int
) -> List[List[int]]:
    seqs: List[List[int]] = []
    for t in texts:
        idx = [vocab_to_index.get(w, 0) for w in jieba.cut(str(t))]
        if len(idx) < text_len:
            idx += [1] * (text_len - len(idx))
        else:
            idx = idx[:text_len]
        seqs.append(idx)
    return seqs