import numpy as np
import os

def pca_probe_csv(data_file_path, center=True, scale=True):
    X = np.loadtxt(data_file_path, delimiter=',', dtype=float)  # shape: (n_samples, n_features)
    n, p = X.shape
    if center:
        mu = X.mean(axis=0)
        X = X - mu
    else:
        mu = np.zeros(p)
    if scale:
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0  # 避免除以 0
        X = X / std
    else:
        std = np.ones(p)

    U, S, VT = np.linalg.svd(X, full_matrices=False)
    ev = (S ** 2) / (n - 1)             
    evr = ev / ev.sum()                
    cum_evr = np.cumsum(evr)           

    k95 = int(np.searchsorted(cum_evr, 0.95) + 1)
    k99 = int(np.searchsorted(cum_evr, 0.99) + 1)

    print(f"样本数 n={n}, 特征数 p={p}")
    print(f"解释 95% 方差所需主成分数: {k95}")
    print(f"解释 99% 方差所需主成分数: {k99}")
    suggested = max(32, min(128, k95))
    print(f"建议起步 latent_dim: {suggested} （可再在 [{max(32, min(k95,256))}, {min(k99,256)}] 范围内微调）")

    return {
        "n": n, "p": p,
        "k95": k95, "k99": k99,
        "evr": evr, "cum_evr": cum_evr,
        "mean": mu, "std": std,
        "suggested_latent": suggested
    }

if __name__ == "__main__":
    trainsize = 500
    month = 'may'
    data_file_path = f'E:\\ukbprotein\\Data\\{trainsize}x_{month}.csv'
    res = pca_probe_csv(data_file_path)
    print(res)
