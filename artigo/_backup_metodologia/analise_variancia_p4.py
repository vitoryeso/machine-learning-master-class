import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

def gen(n, seed):
    rng = np.random.default_rng(seed)
    half = n // 2
    y = np.array([0]*half + [1]*(n-half))
    cx = np.where(y==0, -1.5, 1.5).astype(float)
    cy = np.where(y==0, -1.0, 1.0).astype(float)
    X = np.stack([cx + rng.normal(size=n), cy + rng.normal(size=n)], axis=1)
    perm = rng.permutation(n)
    return X[perm], y[perm]

def acc(n_total, seed, test_frac=0.2):
    X, y = gen(n_total, seed)
    cut = int((1-test_frac)*n_total)
    clf = LogisticRegression(C=1e6).fit(X[:cut], y[:cut])   # ~sem regularizacao, como o P4
    return clf.score(X[cut:], y[cut:])

d = np.sqrt(13); bayes_acc = 1 - norm.cdf(-d/2)
print(f"Bayes: acc ~ {bayes_acc:.4f} (erro {1-bayes_acc:.4f})\n")

print("(A) teste FIXO em 60 pts (n_total=300) — mais seeds NAO encolhe o std:")
print(f"{'n_seeds':>8}{'mean_acc':>11}{'std_acc':>10}")
for ns in [3,5,10,30,100,300]:
    a = [acc(300, 42+i) for i in range(ns)]
    print(f"{ns:>8}{np.mean(a):>11.4f}{np.std(a,ddof=1):>10.4f}")
print(f"  SE binomial teorico @60 pts, p=0.96: {np.sqrt(0.96*0.04/60):.4f}\n")

print("(B) n_seeds=200 fixo — teste MAIOR encolhe o std (~1/sqrt(n_teste)):")
print(f"{'n_total':>8}{'n_teste':>9}{'mean_acc':>11}{'std_acc':>10}{'SE_binom':>10}")
for nt in [300,1000,3000,10000]:
    a = [acc(nt, 1000+i) for i in range(200)]
    nte = nt - int(0.8*nt)
    print(f"{nt:>8}{nte:>9}{np.mean(a):>11.4f}{np.std(a,ddof=1):>10.4f}{np.sqrt(0.96*0.04/nte):>10.4f}")
