import catboost as cb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


def run_cat(X_train, X_test, y_train, categorical_cols=[]):
    y_preds = []
    models = []
    oof_train = np.zeros((len(X_train),))
    cv = GroupKFold(n_splits=5)
    X_test = X_test.drop("mesh20", axis=1)

    params = {
        "depth": 6,
        "learning_rate": 0.1,
        "iterations": 5000,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 777,
        "allow_writing_files": False,
        "task_type": "CPU",
        "early_stopping_rounds": 50,
    }

    for fold_id, (train_index, valid_index) in enumerate(
        cv.split(X_train, groups=X_train["mesh20"])
    ):
        X_tr = X_train.drop("mesh20", axis=1).loc[train_index, :]
        X_val = X_train.drop("mesh20", axis=1).loc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            cat_features=categorical_cols,
            eval_set=(X_val, y_val),
            verbose=100,
            use_best_model=True,
            plot=False,
        )
        oof_train[valid_index] = model.predict(X_val)
        joblib.dump(model, f"lgb_{fold_id}.pkl")
        models.append(model)

        y_pred = model.predict(X_test)
        y_preds.append(y_pred)

    return oof_train, y_preds, models


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
    oof_train, y_preds, models = run_cat(X_train, X_test, y_train, categorical_cols)
    print(mean_squared_error(y_train, np.clip(oof_train, 0, 1), squared=False))
    sub[1] = np.clip(np.average(y_preds, axis=0), 0, 1)
    sub.to_csv("submission.csv", index=False, header=None)
