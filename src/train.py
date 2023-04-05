import os
import logging

import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE, ADASYN
from catboost import Pool, CatBoostClassifier

from utils import load_yaml_config

# Set logging
logging.basicConfig(level=logging.INFO, force=True)


# Loading config file
logging.info(f"Loading the config file")

cfg = load_yaml_config("../config/config.yaml")

# Load the preprocessed data
logging.info(f"Loading processed data")

data_train = pd.read_csv(cfg["data"]["train_data_path"])
X_train = data_train.copy().drop(columns=cfg["data"]["target_variable"])
y_train = data_train[cfg["data"]["target_variable"]]

data_test = pd.read_csv(cfg["data"]["test_data_path"])
X_test = data_test.copy().drop(columns=cfg["data"]["target_variable"])
y_test = data_test[cfg["data"]["target_variable"]]

# Train chosen model
CHOSEN_MODEL = cfg["model"]["model_architecture"]
logging.info(f"Begining training process ...")

# Use balance weight for training since it is imbalanced
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# train logistic regression model
if CHOSEN_MODEL == "logistic_regression":
    logging.info(f"Model chosen for training: logistic_regression")

    # Instantiate logistic regression model
    model = LogisticRegression()

    # Train model
    logging.info(f"Training model")
    model.fit(X_train, y_train)

    # Save trained model
    logging.info(f"Training completed. Saving model")
    SAVE_MODEL_PATH = os.path.join(
        cfg["model"]["save_model_artifacts_folder"], "trained_model.pkl"
    )
    pickle.dump(model, open(SAVE_MODEL_PATH, "wb"))

# train xbgoost model
elif CHOSEN_MODEL == "xgboost":
    logging.info(f"Model chosen for training: xgboost")

    # Instantiate model
    model = XGBClassifier()

    # Fit the train data
    logging.info(f"Training model")
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Save trained model
    logging.info(f"Training completed. Saving model")
    SAVE_MODEL_PATH = os.path.join(
        cfg["model"]["save_model_artifacts_folder"], "trained_model.model"
    )
    model.save_model("trained_model.model")

# train catboost model
elif CHOSEN_MODEL == "catboost":
    logging.info(f"Model chosen for training: catboost")

    # Load the data into pool
    train_pool = Pool(data=X_train, label=y_train)
    test_pool = Pool(data=X_test, label=y_test)

    # Insantiate the catboost model
    model = CatBoostClassifier(
        iterations=cfg["cat_boost_model"]["iterations"],
        learning_rate=cfg["cat_boost_model"]["learning_rate"],
        random_strength=cfg["cat_boost_model"]["random_strength"],
        depth=cfg["cat_boost_model"]["depth"],
        loss_function=cfg["cat_boost_model"]["loss_function"],
        eval_metric=cfg["cat_boost_model"]["eval_metric"],
        leaf_estimation_method=cfg["cat_boost_model"]["leaf_estimation_method"],
    )

    # Fit the train data
    logging.info(f"Training model")
    model.fit(train_pool, plot=True, eval_set=test_pool)

    # Save the model
    logging.info(f"Training completed. Saving model")
    SAVE_MODEL_PATH = os.path.join(
        cfg["model"]["save_model_artifacts_folder"], "trained_model"
    )
    model.save_model(fname=SAVE_MODEL_PATH, format="cbm")


# Evaluate the performane of trained model
logging.info(f"Evaluating trained model: {CHOSEN_MODEL} performance ...")

# Get the pred on test data
pred = model.predict(X_test)
target_names = ["A", "D"]
report = classification_report(
    y_test, pred, target_names=target_names, output_dict=True
)
df_report = pd.DataFrame(report).transpose()
EVAL_RES_PATH = cfg["data"]["final_results_path"]
df_report.to_csv(EVAL_RES_PATH, index=False)

# Calculate the roc score
# auc_score = roc_auc_score(y_test, pred)

logging.info(f"Evaluation complete. Results saved to: {EVAL_RES_PATH}. AUC score: ")

if __name__ == "__main__":
    pass
