import numpy as np
import pandas as pd


def netflix(es, ps, e0, la=0.0001):
    """Combine predictions with the optimal weights to minimize RMSE.
    Args:
        es (list of float): RMSEs of predictions
        ps (list of np.array): predictions
        e0 (float): RMSE of all zero prediction
        la (float): lambda as in the ridge regression
    Returns:
        (tuple):
            - (np.array): ensemble predictions
            - (np.array): weights for input predictions
    """
    m = len(es)
    n = len(ps[0])

    X = np.stack(ps).T
    pTy = 0.5 * (n * e0**2 + (X**2).sum(axis=0) - n * np.array(es) ** 2)

    w = np.linalg.pinv(X.T.dot(X) + la * n * np.eye(m)).dot(pTy)
    return X.dot(w), w


if __name__ == "__main__":

    sub001 = pd.read_csv("submission_lgbm_baseline.csv", header=None)[1].values
    sub002 = pd.read_csv("submission_lgbm_feat_select.csv", header=None)[1].values
    sub003 = pd.read_csv("submission_cat_feat_select.csv", header=None)[1].values

    es = [
        0.1843373,
        0.1976116,
        0.1921030,
    ]
    ps = [
        sub003,
        sub002,
        sub001,
    ]
    e0 = 0.1998473

    pred, w = netflix(es, ps, e0, la=0.001)
    print(w)

    exp_name = "netflix000"
    sub = pd.read_csv("../input/jsai2023/submit_example.csv", header=None)
    sub[1] = np.clip(pred, 0, 1)
    sub.to_csv(f"submission_{exp_name}.csv", index=False, header=None)
