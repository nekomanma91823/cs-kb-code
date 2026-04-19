import math
from collections import Counter


def tf(term: str, doc: list[str]) -> float:
    counts = Counter(doc)  # 文書内で各単語が何回出るかを数える
    return counts[term] / len(doc)  # TF: 単語termの出現回数 / 文書中の総単語数


def idf(term: str, corpus: list[list[str]]) -> float:
    n_docs = len(corpus)  # コーパス内の総文書数N
    df = sum(1 for doc in corpus if term in doc)  # termを含む文書数df
    return (
        math.log((n_docs + 1) / (df + 1)) + 1
    )  # IDF: log((N+1)/(df+1))+1（0割防止の平滑化）


def tfidf(term: str, doc: list[str], corpus: list[list[str]]) -> float:
    return tf(term, doc) * idf(term, corpus)  # TF-IDF: tf × idf


def tfidf_vector(doc: list[str], corpus: list[list[str]]) -> dict[str, float]:
    vocab = set(w for d in corpus for w in d)  # コーパス全体の語彙集合（特徴次元）
    vec = {
        term: tfidf(term, doc, corpus) for term in vocab
    }  # 各語のTF-IDF値でベクトル化
    norm = math.sqrt(sum(v**2 for v in vec.values()))  # L2ノルム: sqrt(Σ v_i^2)
    return (
        {term: v / norm for term, v in vec.items()} if norm > 0 else vec
    )  # 正規化後ベクトル（長さ1）


def cosine_similarity(v1: dict, v2: dict) -> float:
    keys = set(v1) & set(v2)  # 共通次元（共通語）
    return sum(
        v1[k] * v2[k] for k in keys
    )  # コサイン類似度: 正規化済みベクトル同士の内積


# 使用例
corpus = [
    ["機械", "学習", "で", "モデル", "を", "学習", "する"],
    ["深層", "学習", "は", "機械", "学習", "の", "一種"],
    ["自然", "言語", "処理", "で", "テキスト", "を", "分析"],
]
v0 = tfidf_vector(corpus[0], corpus)
v1 = tfidf_vector(corpus[1], corpus)
v2 = tfidf_vector(corpus[2], corpus)
print(cosine_similarity(v0, v1))  # 文書0と1の類似度（高いはず）
print(cosine_similarity(v0, v2))  # 文書0と2の類似度（低いはず）
