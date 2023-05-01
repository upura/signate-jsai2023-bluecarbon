import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


def run_lgbm(X_train, X_test, y_train, categorical_cols=[]):
    y_preds = []
    models = []
    oof_train = np.zeros((len(X_train),))
    cv = GroupKFold(n_splits=5)
    X_test = X_test.drop("mesh20", axis=1)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 32,
        "max_depth": 4,
        "feature_fraction": 0.8,
        "subsample_freq": 1,
        "bagging_fraction": 0.7,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "boosting": "gbdt",
        "lambda_l1": 0.4,
        "lambda_l2": 0.4,
        "verbosity": -1,
        "random_state": 42,
    }

    for fold_id, (train_index, valid_index) in enumerate(
        cv.split(X_train, groups=X_train["mesh20"])
    ):
        X_tr = X_train.drop("mesh20", axis=1).loc[train_index, :]
        X_val = X_train.drop("mesh20", axis=1).loc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)

        lgb_eval = lgb.Dataset(
            X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols
        )

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        oof_train[valid_index] = model.predict(
            X_val, num_iteration=model.best_iteration
        )
        joblib.dump(model, f"lgb_{fold_id}.pkl")
        models.append(model)

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_preds.append(y_pred)

    return oof_train, y_preds, models


def visualize_importance(models, X_train):
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importance()
        _df["column"] = X_train.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, _df], axis=0, ignore_index=True
        )

    order = (
        feature_importance_df.groupby("column")
        .sum()[["feature_importance"]]
        .sort_values("feature_importance", ascending=False)
        .index[:50]
    )

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.4), 7))
    sns.boxenplot(
        data=feature_importance_df,
        x="column",
        y="feature_importance",
        order=order,
        ax=ax,
        palette="viridis",
    )
    ax.tick_params(axis="x", rotation=90)
    ax.grid()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":

    train = pd.read_csv("../input/jsai2023/train_data.csv")
    test = pd.read_csv("../input/jsai2023/test_data.csv")
    sub = pd.read_csv("../input/jsai2023/submit_example.csv", header=None)
    print(train.shape, test.shape, sub.shape)

    train["mesh20_0"] = train["mesh20"].str.split("_").map(lambda x: x[0]).astype(int)
    train["mesh20_1"] = train["mesh20"].str.split("_").map(lambda x: x[1]).astype(int)
    test["mesh20_0"] = test["mesh20"].str.split("_").map(lambda x: x[0]).astype(int)
    test["mesh20_1"] = test["mesh20"].str.split("_").map(lambda x: x[1]).astype(int)

    target_col = "cover"

    use_cols = [
        "depth_original",
        "area",
        "hist_warm_sst",
        "warm_sst",
        "MIN_GARI",
        "Date_Acquired",
        "depth",
        "sst_diff",
        "month",
        "fetch",
        "MIN_D678_500",
        "year",
        "lat",
        "MIN_IF_2017",
        "sst_ymd",
        "river_dist",
        "MIN_RDVI_2002",
        "MED_VARIgreen_2004",
        "MIN_Blue_2016",
        "MAX_TIRS1_2013",
        "MED_H_2019",
        "MAX_RDVI_2003",
        "MAX_NDFI2",
        "EVI",
        "MIN_RDVI_2006",
        "TIRS2",
        "MED_GARI",
        "mesh20",
    ]

    X_train = train[use_cols]
    X_test = test[use_cols]
    y_train = train[target_col]

    categorical_cols = []
    oof_train, y_preds, models = run_lgbm(X_train, X_test, y_train, categorical_cols)
    visualize_importance(models, X_train.drop("mesh20", axis=1))
    print(mean_squared_error(y_train, np.clip(oof_train, 0, 1), squared=False))
    sub[1] = np.clip(np.average(y_preds, axis=0), 0, 1)
    sub.to_csv("submission_lgbm_baseline.csv", index=False, header=None)
