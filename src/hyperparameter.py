import logging

import yaml
import pandas as pd

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report,
    roc_curve,
    precision_recall_curve,
    f1_score,
)
from imblearn.over_sampling import SMOTE, ADASYN
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from catboost import Pool, CatBoostClassifier

# Set logging
logging.basicConfig(level=logging.INFO, force=True)


def objective(
    space: dict,
    train_data_features: pd.DataFrame,
    train_data_target: list,
    test_data_features: pd.DataFrame,
    test_data_target: pd.DataFrame,
):
    """Define the objective function for hyperopt hyperparameter tuning of xgboost model

    Args:
        space (dict): search space

    Returns:
        dict: The f score and status as dict
    """
    clf = XGBClassifier(
        n_estimators=space["n_estimators"],
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        reg_alpha=int(space["reg_alpha"]),
        min_child_weight=int(space["min_child_weight"]),
        colsample_bytree=int(space["colsample_bytree"]),
    )

    evaluation = [
        (train_data_features, train_data_target),
        (test_data_features, test_data_target),
    ]

    clf.fit(
        train_data_features,
        train_data_target,
        eval_set=evaluation,
        eval_metric="auc",
        early_stopping_rounds=10,
        verbose=False,
    )

    pred = clf.predict(X_test)
    f_score = f1_score(y_test, pred)
    print("SCORE:", f_score)
    return {"loss": -f_score, "status": STATUS_OK}


# # If tune hyperparameter
# if cfg["hyperparameter"]["tune_hyperparameter"]:
#     logging.info("Hyperparameter tuning of xgboost model")

#     # define search space
#     space = {
#         "max_depth": hp.quniform(
#             "max_depth",
#             cfg["hyperparameter"]["xgboost"]["min"],
#             cfg["hyperparameter"]["xgboost"]["max"],
#             cfg["hyperparameter"]["xgboost"]["step_size"],
#         ),
#         "gamma": hp.uniform(
#             "gamma",
#             cfg["hyperparameter"]["xgboost"]["min"],
#             cfg["hyperparameter"]["xgboost"]["max"],
#         ),
#         "reg_alpha": hp.quniform(
#             "reg_alpha",
#             cfg["hyperparameter"]["xgboost"]["min"],
#             cfg["hyperparameter"]["xgboost"]["max"],
#             cfg["hyperparameter"]["xgboost"]["step_size"],
#         ),
#         "reg_lambda": hp.uniform(
#             "reg_lambda",
#             cfg["hyperparameter"]["xgboost"]["min"],
#             cfg["hyperparameter"]["xgboost"]["max"],
#         ),
#         "colsample_bytree": hp.uniform(
#             "colsample_bytree",
#             cfg["hyperparameter"]["xgboost"]["min"],
#             cfg["hyperparameter"]["xgboost"]["max"],
#         ),
#         "min_child_weight": hp.quniform(
#             "min_child_weight",
#             cfg["hyperparameter"]["xgboost"]["max"],
#             cfg["hyperparameter"]["xgboost"]["min"],
#             cfg["hyperparameter"]["xgboost"]["step_size"],
#         ),
#         "n_estimators": cfg["xgboost"]["n_estimators"],
#         "seed": cfg["random_seed"],
#     }

# # Instantiate the trials
# trials = Trials()

# # Define the best params
# best_hyperparams = fmin(
#     fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials
# )
