import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# 画像を保存するディレクトリを作成
output_dir = "math_and_stats/probability/images"
os.makedirs(output_dir, exist_ok=True)


def save_plot(filename, title):
    # 日本語フォント設定
    plt.rcParams['font.family'] = 'MS Gothic'

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.subplots_adjust(bottom=0.1)

    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# ==========================================
# 【離散型確率分布】
# ==========================================


# 離散一様分布
# 説明: サイコロの目のように、有限の事象が等確率で起こる場合の確率分布
# 数式: P(X=k) = 1/n
def plot_discrete_uniform():
    x = np.arange(1, 7)
    y = np.ones(6) / 6
    plt.bar(x, y, alpha=0.7)
    save_plot("01_discrete_uniform.png", "離散一様分布")


# ベルヌーイ分布
# 説明: 結果が2通り（成功・失敗）しかない試行の、成功回数が従う分布
# 数式: P(X=x) = p^x (1-p)^{1-x} (x ∈ {0, 1})
def plot_bernoulli():
    p = 0.3
    x = [0, 1]
    y = [1 - p, p]
    plt.bar(x, y, alpha=0.7)
    plt.xticks(x, ['Fail(0)', 'Success(1)'])
    save_plot("02_bernoulli.png", f"ベルヌーイ分布 (p={p})")


# 二項分布
# 説明: 独立なベルヌーイ試行をn回繰り返したときの成功回数が従う分布
# 数式: P(X=k) = C(n,k) p^k (1-p)^{n-k}
def plot_binomial():
    n, p = 10, 0.5
    x = np.arange(0, n + 1)
    y = stats.binom.pmf(x, n, p)
    plt.bar(x, y, alpha=0.7)
    save_plot("03_binomial.png", f"二項分布 (n={n}, p={p})")


# 幾何分布
# 説明: 独立なベルヌーイ試行で初めて成功するまでの試行回数が従う分布
# 数式: P(X=k) = (1-p)^{k-1}p
def plot_geometric():
    p = 0.3
    x = np.arange(1, 11)
    y = stats.geom.pmf(x, p)
    plt.bar(x, y, alpha=0.7)
    save_plot("04_geometric.png", f"幾何分布 (p={p})")


# 負の二項分布
# 説明: r回目に成功するまでに失敗した回数が従う分布
# 数式: P(X=k) = C(k+r-1, k) p^r (1-p)^k
def plot_negative_binomial():
    n, p = 5, 0.5
    x = np.arange(0, 15)
    y = stats.nbinom.pmf(x, n, p)
    plt.bar(x, y, alpha=0.7)
    save_plot("05_negative_binomial.png", f"負の二項分布 (r={n}, p={p})")


# ポアソン分布
# 説明: 特定単位（時間・空間等）に事象が発生する回数をモデル化する分布
# 数式: P(X=k) = (λ^k e^{-λ}) / k!
def plot_poisson():
    mu = 3
    x = np.arange(0, 11)
    y = stats.poisson.pmf(x, mu)
    plt.bar(x, y, alpha=0.7)
    save_plot("06_poisson.png", f"ポアソン分布 (λ={mu})")


# 多項分布
# 説明: 結果が3通り以上ある多項試行を独立に繰り返した分布
# 数式: P(x_1, ..., x_k) = (n! / (x_1!...x_k!)) p_1^{x_1}...p_k^{x_k}
def plot_multinomial():
    p = [0.2, 0.3, 0.5]
    x = ['Cat 1(p=0.2)', 'Cat 2(p=0.3)', 'Cat 3(p=0.5)']
    y = [10 * pi for pi in p]
    plt.bar(x, y, alpha=0.7)
    save_plot("07_multinomial.png", "多項分布")


# 超幾何分布
# 説明: N個中M個が当たりの状態から、非復元でn個引いたときの当たりの数が従う分布
# 数式: P(X=k) = (C(M, k) C(N-M, n-k)) / C(N, n)
def plot_hypergeometric():
    M, n, N = 20, 7, 12
    x = np.arange(0, n + 1)
    y = stats.hypergeom.pmf(x, M, n, N)
    plt.bar(x, y, alpha=0.7)
    save_plot("08_hypergeometric.png", f"超幾何分布 (N={M}, M={n}, n={N})")


# ==========================================
# 【連続型確率分布】
# ==========================================


# 連続一様分布
# 説明: ある区間内で常に一定の確率密度を持つ分布
# 数式: f(x) = 1/(b-a) (a <= x <= b)
def plot_continuous_uniform():
    x = np.linspace(0, 10, 100)
    y = stats.uniform.pdf(x, loc=2, scale=6)  # [2, 8]
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("09_continuous_uniform.png", "連続一様分布 [2, 8]")


# 正規分布
# 説明: 期待値と分散によって定まる釣鐘型の分布。自然界や統計学で広く見られる
# 数式: f(x) = (1/sqrt(2πσ^2)) exp(-(x-μ)^2 / 2σ^2)
def plot_normal():
    x = np.linspace(-10, 10, 100)
    y = stats.norm.pdf(x, loc=2, scale=3)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("10_normal.png", "正規分布 (μ=2, σ=3)")


# 標準正規分布
# 説明: 平均が0、分散が1である特別な正規分布
# 数式: f(x) = (1/sqrt(2π)) exp(-x^2 / 2)
def plot_standard_normal():
    x = np.linspace(-4, 4, 100)
    y = stats.norm.pdf(x, loc=0, scale=1)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("11_standard_normal.png", "標準正規分布 (μ=0, σ=1)")


# 対数正規分布
# 説明: 対数を取った変数が正規分布に従うような確率変数の分布
# 数式: f(x) = (1/(x σ sqrt(2π))) exp(-(ln x - μ)^2 / 2σ^2)
def plot_lognormal():
    x = np.linspace(0, 5, 100)
    y = stats.lognorm.pdf(x, s=0.5, scale=np.exp(0))
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("12_lognormal.png", "対数正規分布")


# 指数分布
# 説明: ランダムなイベントが次に起こるまでの待機時間を表す分布。無記憶性を持つ
# 数式: f(x) = λ e^{-λ x} (x >= 0)
def plot_exponential():
    x = np.linspace(0, 5, 100)
    y = stats.expon.pdf(x, scale=1)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("13_exponential.png", "指数分布")


# ガンマ分布
# 説明: ランダムなイベントが複数回起こるまでの待機時間を表す分布。指数分布の一般化
# 数式: f(x) = (x^{k-1} e^{-x/θ}) / (Γ(k) θ^k)
def plot_gamma():
    x = np.linspace(0, 20, 100)
    y = stats.gamma.pdf(x, a=2, scale=2)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("14_gamma.png", "ガンマ分布 (形状母数2, 尺度母数2)")


# ベータ分布
# 説明: 区間[0,1]で定義され、確率や比率をモデル化するのに適した分布
# 数式: f(x) = (x^{α-1}(1-x)^{β-1}) / B(α, β)
def plot_beta():
    x = np.linspace(0, 1, 100)
    y = stats.beta.pdf(x, a=2, b=5)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("15_beta.png", "ベータ分布 (α=2, β=5)")


# コーシー分布
# 説明: 平均（期待値）や分散が存在しない、裾が厚い分布
# 数式: f(x) = 1 / (πγ [1 + ((x-x_0)/γ)^2])
def plot_cauchy():
    x = np.linspace(-5, 5, 100)
    y = stats.cauchy.pdf(x, loc=0, scale=1)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("16_cauchy.png", "コーシー分布")


# ==========================================
# 【標本分布】
# ==========================================


# カイ二乗分布
# 説明: 標準正規分布に従う独立なk個の確率変数の2乗和が従う分布
# 数式: f(x) = (1 / (2^{k/2}Γ(k/2))) x^{k/2-1} e^{-x/2}
def plot_chi2():
    x = np.linspace(0, 20, 100)
    y = stats.chi2.pdf(x, df=5)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("17_chi_square.png", "カイ二乗分布 (df=5)")


# 非心カイ二乗分布
# 説明: 標準正規分布に従う変数に定数を加え、2乗した和が従う分布
# 数式: 非心パラメータ λ を持つ
def plot_ncx2():
    x = np.linspace(0, 20, 100)
    y = stats.ncx2.pdf(x, df=5, nc=2)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("18_non_central_chi_square.png", "非心カイ二乗分布")


# t分布
# 説明: 母分散が未知の場合の、平均の仮説検定や区間推定に利用される分布
# 数式: f(t) = (Γ((ν+1)/2) / sqrt(νπ)Γ(ν/2)) (1+t^2/ν)^{-(ν+1)/2}
def plot_t():
    x = np.linspace(-4, 4, 100)
    y = stats.t.pdf(x, df=5)
    plt.plot(x, y, label='t (df=5)')
    plt.plot(x, stats.norm.pdf(x), linestyle='--', label='Standard Normal')
    plt.legend(loc='upper right')
    plt.fill_between(x, y, alpha=0.3)
    save_plot("19_t_distribution.png", "t分布 (df=5)")


# F分布
# 説明: 2つの独立なカイ二乗分布に従う変数の比が従う、分散の比較等に使う分布
# 数式: f(x) = (1/B(d_1/2, d_2/2)) (d_1/d_2)^{d_1/2} x^{d_1/2-1} (1+(d_1/d_2)x)^{-(d_1+d_2)/2}
def plot_f():
    x = np.linspace(0, 5, 100)
    y = stats.f.pdf(x, dfn=5, dfd=2)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("20_f_distribution.png", "F分布")


# 非心F分布
# 説明: 非心カイ二乗分布とカイ二乗分布に従う変数から定義される分布
# 数式: 非定数成分が存在する分散比について用いられる
def plot_ncf():
    x = np.linspace(0, 5, 100)
    y = stats.ncf.pdf(x, dfn=5, dfd=2, nc=2)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    save_plot("21_non_central_f.png", "非心F分布")


# ==========================================
# 【多変量分布・その他】
# ==========================================


# 多変量正規分布
# 説明: 複数の確率変数の組が従う正規分布。平均ベクトルと分散共分散行列で定まる
# 数式: f(x) = (1/sqrt((2π)^k |Σ|)) exp(-0.5(x-μ)^T Σ^{-1}(x-μ))
def plot_multivariate_normal():
    x, y = np.mgrid[-3:3:0.05, -3:3:0.05]
    pos = np.dstack((x, y))
    rv = stats.multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    plt.contourf(x, y, rv.pdf(pos), cmap='viridis')
    plt.colorbar()
    save_plot("22_multivariate_normal.png", "多変量正規分布")


# 2変量標準正規分布
# 説明: 2変量正規分布のうち、独立で平均0、分散1の分布
# 数式: μ = (0, 0)^T, Σ = I
def plot_bivariate_standard_normal():
    x, y = np.mgrid[-3:3:0.05, -3:3:0.05]
    pos = np.dstack((x, y))
    rv = stats.multivariate_normal([0, 0], [[1, 0], [0, 1]])
    plt.contourf(x, y, rv.pdf(pos), cmap='viridis')
    plt.colorbar()
    save_plot("23_bivariate_standard_normal.png", "2変量標準正規分布")


# 混合正規分布
# 説明: 複数の正規分布の確率密度関数を一定の割合で足し合わせた分布
# 数式: f(x) = Σ w_i N(x | μ_i, σ_i^2)
def plot_gaussian_mixture():
    x = np.linspace(-5, 10, 1000)
    y1 = 0.4 * stats.norm.pdf(x, loc=0, scale=1)
    y2 = 0.6 * stats.norm.pdf(x, loc=5, scale=1.5)
    plt.plot(x, y1, '--', alpha=0.5, label='Comp 1')
    plt.plot(x, y2, '--', alpha=0.5, label='Comp 2')
    plt.plot(x, y1 + y2, label='Mixture')
    plt.legend(loc='upper right')
    save_plot("24_gaussian_mixture.png", "混合正規分布")


# 指数型分布族
# 説明: 確率密度関数が特定の指数関数の形式で表される確率分布の総称
# 数式: f(x|θ) = h(x) exp(η(θ)・T(x) - A(θ))
def plot_exponential_family():
    x = np.arange(0, 15)
    for mu in [1, 4, 10]:
        plt.plot(x, stats.poisson.pmf(x, mu), marker='o', label=f'Poisson(λ={mu})')
    plt.legend()
    save_plot("25_exponential_family.png", "指数型分布族 (例:ポアソン)")


if __name__ == "__main__":
    plot_discrete_uniform()
    plot_bernoulli()
    plot_binomial()
    plot_geometric()
    plot_negative_binomial()
    plot_poisson()
    plot_multinomial()
    plot_hypergeometric()
    plot_continuous_uniform()
    plot_normal()
    plot_standard_normal()
    plot_lognormal()
    plot_exponential()
    plot_gamma()
    plot_beta()
    plot_cauchy()
    plot_chi2()
    plot_ncx2()
    plot_t()
    plot_f()
    plot_ncf()
    plot_multivariate_normal()
    plot_bivariate_standard_normal()
    plot_gaussian_mixture()
    plot_exponential_family()
    print("生成完了しました！")
