import math


def build_cooccurrence_matrix(corpus: list[list[str]], window: int = 2):
    vocab = sorted(set(w for sent in corpus for w in sent))
    w2i = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    M = [[0] * n for _ in range(n)]

    for sent in corpus:
        for t, word in enumerate(sent):
            start = max(0, t - window)
            end = min(len(sent), t + window + 1)
            for ctx in sent[start:end]:
                if ctx != word:
                    M[w2i[word]][w2i[ctx]] += 1
    return vocab, M


def ppmi(M: list[list[float]]):
    total = sum(sum(row) for row in M)
    row_sums = [sum(row) for row in M]
    col_sums = [sum(M[i][j] for i in range(len(M))) for j in range(len(M[0]))]
    n = len(M)
    result = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if M[i][j] == 0:
                continue
            pij = M[i][j] / total
            pi = row_sums[i] / total
            pj = col_sums[j] / total
            pmi = math.log(pij / (pi * pj))
            result[i][j] = max(0.0, pmi)
    return result


# 使用例
corpus = [
    ["銀行", "に", "預金", "した"],
    ["銀行", "の", "融資", "を", "受けた"],
    ["預金", "と", "融資", "は", "銀行", "の", "業務"],
]
vocab, M = build_cooccurrence_matrix(corpus, window=2)
P = ppmi(M)
print("Vocabulary:", vocab)
print("Co-occurrence Matrix:", M)
print("PPMI Matrix:", P)
