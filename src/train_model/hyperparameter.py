import logging

import pandas as pd

from xgboost import XGBClassifier

from hyperopt import STATUS_OK, Trials, fmin, tpe
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Set logging
logging.basicConfig(level=logging.INFO, force=True)


def tune_hyerparameters(
    model_type: str,
    search_space: dict,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
):
    """Get the optimial hyperparameter combination

    Args:
        model_type (str): Model type selected
        search_space (dict): Hyperparameter search space
        features_train (pd.DataFrame): Features from train split
        features_test (pd.DataFrame): Features from test split
        target_train (pd.DataFrame): Target from train split
        target_test (pd.DataFrame): Target from test split

    Returns:
        dict: Optimal hyperparameter configuration
    """
    # Check the model type
    if model_type == "xgboost":
        obj = objective(
            model_type=model_type,
            search_space=search_space,
            features_train=features_train,
            features_test=features_test,
            target_train=target_train,
            target_test=target_test,
        )

    elif model_type == "catboost":
        obj = objective(
            model_type=model_type,
            search_space=search_space,
            features_train=features_train,
            features_test=features_test,
            target_train=target_train,
            target_test=target_test,
        )

    # Start hyperparameter tuning for 100 trials
    trials = Trials()

    best_hyperparams = fmin(
        fn=obj, space=search_space, algo=tpe.suggest, max_evals=10, trials=trials
    )

    return best_hyperparams


# Define the objective sweep function using xgboost
def objective(
    model_type: str,
    search_space: dict,
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
):
    """The objective function for hyperparameter tuning

    Args:
        model_type (str): Model type selected
        search_space (dict): Hyperparameter search space
        features_train (pd.DataFrame): Features from train split
        features_test (pd.DataFrame): Features from test split
        target_train (pd.DataFrame): Target from train split
        target_test (pd.DataFrame): Target from test split

    Returns:
        _type_: _description_
    """
    # if xgboost
    if model_type == "xgboost":
        clf = XGBClassifier(
            n_estimators=search_space["n_estimators"],
            max_depth=int(search_space["max_depth"]),
            gamma=search_space["gamma"],
            reg_alpha=int(search_space["reg_alpha"]),
            min_child_weight=int(search_space["min_child_weight"]),
            colsample_bytree=int(search_space["colsample_bytree"]),
        )

        evaluation = [(features_train, target_train), (features_test, target_test)]

        clf.fit(
            features_train,
            target_train,
            eval_set=evaluation,
            eval_metric="auc",
            early_stopping_rounds=10,
            verbose=False,
        )
    # if catboost
    if model_type == "catboost":
        clf = CatBoostClassifier(
            iterations=1000,
            learning_rate=search_space["learning_rate"],
            random_strength=search_space["random_strength"],
            depth=search_space["depth"],
            loss_function="MultiClass",
            eval_metric="AUC",
            leaf_estimation_method="Newton",
        )
        train_pool = Pool(data=features_train, label=target_train)
        test_pool = Pool(data=features_test, label=target_test)

        evaluation = [train_pool, train_pool]

        clf.fit(train_pool, plot=True, eval_set=test_pool)

    # Get the prediction
    pred = clf.predict(features_test)

    # Get the probability of prediction
    y_score = clf.predict_proba(features_test)[:, 1]

    # Calculate the roc score
    roc_auc = roc_auc_score(target_test, y_score)
    print("SCORE:", roc_auc)

    return {"loss": -roc_auc, "status": STATUS_OK}
