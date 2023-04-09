import os
import logging

import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE, ADASYN
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt

# Set logging
logging.basicConfig(level=logging.INFO, force=True)


def oversample_minority_class(
    sampling_method: str,
    features_train: pd.DataFrame,
    target_train: pd.DataFrame,
    random_seed: int,
):
    # Check which sampling method is used
    logging.info(f"Oversampling method selected: SMOTE")
    if sampling_method == "SMOTE":
        X_train, y_train = SMOTE(random_state=random_seed).fit_resample(
            features_train, target_train
        )
    logging.info(f"Oversampling method selected: ADASYN")
    if sampling_method == "ADASYN":
        X_train, y_train = ADASYN(random_state=random_seed).fit_resample(
            features_train, target_train
        )
    logging.info(f"Oversampling method selected: None")
    if sampling_method == "none":
        X_train = features_train
        y_train = target_train

    return X_train, y_train


def train_logistic_regression_model(
    features_train: pd.DataFrame, target_train: pd.DataFrame
):
    """Train logistiic regression model

    Args:
        features_train (pd.DataFrame): Features from train split
        target_train (pd.DataFrame): Target from train split
    """
    # Instantiate logistic regression model
    model = LogisticRegression()

    # Train model
    model.fit(features_train, target_train)
    logging.info(f"Training logistic regression model complete")

    return model


def train_xgboost_model(
    features_train: pd.DataFrame, target_train: pd.DataFrame, weights
):
    """Train xbgoost model

    Args:
        features_train (pd.DataFrame): Features from train split
        target_train (pd.DataFrame): Target from train split
        weights (_type_): Model weights for imbalance class problems
    """
    # Instantiate model
    model = XGBClassifier()

    # Fit the train data
    model.fit(features_train, target_train, sample_weight=weights)
    logging.info(f"Training xbgoost model complete")

    return model


def train_catboost_model(
    train_pool,
    test_pool,
    iterations: int = 1000,
    learning_rate: float = 0.1,
    random_strength: float = 0.1,
    depth: int = 8,
    loss_function: str = "MultiClass",
    eval_metric: str = "AUC",
    leaf_estimation_method: str = "Newton",
    random_seed: int = 2023,
):
    """Train catboost model

    Args:
        train_pool: The train data
        test_pool: The test data
        iterations (int, optional): The max number of trees. Defaults to 1000.
        learning_rate (float, optional): The learning rate used for reducing gradient step. Defaults to 0.1.
        random_strength (float, optional): The amount of randomness to use for scoring splits when the tree structure is selected. Used to avoid overfitting. Defaults to 0.1.
        depth (int, optional): Depth of each tree. Defaults to 8.
        loss_function (str, optional): The loss function to use. Defaults to "MultiClass".
        eval_metric (str, optional): The evaluation metric to use. Defaults to "AUC".
        leaf_estimation_method (str, optional): The method used to calculate the values in leaves. Defaults to "Newton".
        random_seed (int, optional): The randomm seed to use. Defaults to 2023.
    """
    # Insantiate the catboost model
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        random_strength=random_strength,
        depth=depth,
        loss_function=loss_function,
        eval_metric=eval_metric,
        leaf_estimation_method=leaf_estimation_method,
        random_seed=random_seed,
    )

    # Fit the train data
    model.fit(train_pool, plot=True, eval_set=test_pool)
    logging.info(f"Training catboost model complete")

    return model


def save_trained_model(model_type: str, trained_model, save_model_path: str):
    """Save the trained model

    Args:
        model_type (str): Model architecture selected. Currently supports only logistic_regression, xgboost, catboost
        trained_model (_type_): Trained model
        save_model_path (str): Path to save the model
    """
    # Save logistic regression model
    if model_type == "logistic_regression":
        MODEL_PATH = os.path.join(save_model_path, "trained_model.pkl")
        pickle.dump(trained_model, open(MODEL_PATH, "wb"))

    # Save xgboost model
    elif model_type == "xgboost":
        MODEL_PATH = os.path.join(save_model_path, "trained_model.model")
        trained_model.save_model(MODEL_PATH)

    # Save catboost model
    elif model_type == "catboost":
        MODEL_PATH = os.path.join(save_model_path, "trained_model")
        trained_model.save_model(fname=MODEL_PATH, format="cbm")

    logging.info("Model saved and can be found here: {MODEL_PATH}")


def get_classification_report(
    train_model,
    features_test: pd.DataFrame,
    target_test: pd.DataFrame,
    save_results_path: str,
):
    """Get the classification report

    Args:
        train_model (_type_): Trained model
        features_test (pd.DataFrame): Features from test split
        target_test (pd.DataFrame): Target from test split
        save_results_path (str): Path to save classification report in csv format
    """
    # Get the pred on test data
    pred = train_model.predict(features_test)

    # Get the classification report
    target_names = ["A", "D"]
    report = classification_report(
        target_test, pred, target_names=target_names, output_dict=True
    )

    # Save the report as csv
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(save_results_path, index=False)


def get_auc_score(
    trained_model,
    features_test: pd.DataFrame,
    target_test: pd.DataFrame,
    save_roc_plot_path: str,
):
    """Calculate the auc score and save roc curve

    Args:
        trained_model (_type_): Trained model
        features_test (pd.DataFrame): Features from test split
        target_test (pd.DataFrame): Target from test split
        save_roc_plot_path (str): Path to save the roc curve

    Returns:
        _type_: Auc score
    """
    # Get the preds probability
    y_pred = trained_model.predict_proba(features_test)[:, 1]

    # Get the false positive rate, true positive rate and threshold
    fp_r, tp_r, t = roc_curve(target_test, y_pred)

    # Get the auc score
    auc_score = auc(fp_r, tp_r)

    # Plot the roc curve and save
    plt.figure(figsize=(8, 6))
    plt.plot(fp_r, tp_r, label="AUC = %.2f" % auc_score)
    plt.plot([0, 1], [0, 1], "r--")
    plt.ylabel("TP rate")
    plt.xlabel("FP rate")
    plt.legend(loc=4)
    plt.title("ROC Curve")
    plt.savefig(save_roc_plot_path)

    # return the auc score
    return auc_score
