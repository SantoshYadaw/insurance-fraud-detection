import os
import sys
import inspect
import logging

import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from catboost import Pool, CatBoostClassifier
from hyperopt import hp

# Append path to parent directory (src) for easier import
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from src.utils import load_yaml_config
from train_utils import (
    oversample_minority_class,
    train_logistic_regression_model,
    train_xgboost_model,
    train_catboost_model,
    save_trained_model,
    get_classification_report,
    get_auc_score,
)
from hyperparameter import tune_hyerparameters

# Set logging
logging.basicConfig(level=logging.INFO, force=True)


# Loading config file
logging.info(f"Loading the config file")
# cfg = load_yaml_config("../config/config.yaml")
CFG_PATH = os.path.join(parentdir, "config", "config.yaml")
logging.info("CFG_PATH: {CFG_PATH}")
cfg = load_yaml_config(CFG_PATH)

# Load the preprocessed data
logging.info(f"Loading processed data")

# Train data
data_train = pd.read_csv(cfg["data"]["train_data_path"])
X_train = data_train.copy().drop(columns=cfg["data"]["target_variable"])
y_train = data_train[cfg["data"]["target_variable"]]

# Test data
data_test = pd.read_csv(cfg["data"]["test_data_path"])
X_test = data_test.copy().drop(columns=cfg["data"]["target_variable"])
y_test = data_test[cfg["data"]["target_variable"]]

# Train chosen model
CHOSEN_MODEL = cfg["model"]["model_architecture"]
logging.info(f"Begining training process ...")

# Use balance weight for training since it is imbalanced
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# Use SMOTE or ADASYN for over sampling method of minority class
X_train, y_train = oversample_minority_class(
    sampling_method=cfg["oversampling_method"],
    features_train=X_train,
    target_train=y_train,
    random_seed=cfg["random_seed"],
)

# train logistic regression model
if CHOSEN_MODEL == "logistic_regression":
    logging.info(f"Model chosen for training: logistic_regression")

    # Train logistic regression model
    model = train_logistic_regression_model(
        features_train=X_train, target_train=y_train
    )

# train xbgoost model
elif CHOSEN_MODEL == "xgboost":
    logging.info(f"Model chosen for training: xgboost")

    if cfg["tune_hyperparameters"]:
        logging.info(f"Tuning xgboost hyperparameters")
        # Define the search space
        space = {
            "max_depth": hp.quniform(
                "max_depth",
                cfg["hyperparameters"]["xgboost"]["max_depth"]["min"],
                cfg["hyperparameters"]["xgboost"]["max_depth"]["max"],
                cfg["hyperparameters"]["xgboost"]["max_depth"]["step_size"],
            ),
            "gamma": hp.uniform(
                "gamma",
                cfg["hyperparameters"]["xgboost"]["gamma"]["min"],
                cfg["hyperparameters"]["xgboost"]["gamma"]["max"],
            ),
            "reg_alpha": hp.quniform(
                "reg_alpha",
                cfg["hyperparameters"]["xgboost"]["reg_alpha"]["min"],
                cfg["hyperparameters"]["xgboost"]["reg_alpha"]["max"],
                cfg["hyperparameters"]["xgboost"]["reg_alpha"]["step_size"],
            ),
            "reg_lambda": hp.uniform(
                "reg_lambda",
                cfg["hyperparameters"]["xgboost"]["reg_lambda"]["min"],
                cfg["hyperparameters"]["xgboost"]["reg_alpha"]["max"],
            ),
            "colsample_bytree": hp.uniform(
                "colsample_bytree",
                cfg["hyperparameters"]["xgboost"]["colsample_bytree"]["min"],
                cfg["hyperparameters"]["xgboost"]["reg_alpha"]["max"],
            ),
            "min_child_weight": hp.quniform(
                "min_child_weight",
                cfg["hyperparameters"]["xgboost"]["min_child_weight"]["min"],
                cfg["hyperparameters"]["xgboost"]["reg_alpha"]["max"],
                cfg["hyperparameters"]["xgboost"]["reg_alpha"]["step_size"],
            ),
            "n_estimators": cfg["hyperparameters"]["xgboost"]["n_estimators"],
            "seed": cfg["random_seed"],
        }
        best_params = tune_hyerparameters(
            model_type=CHOSEN_MODEL,
            search_space=space,
            features_train=X_train,
            features_test=X_test,
            target_train=y_train,
            target_test=y_test,
        )
        logging.info(f"Tuning xgboost model using hyperparameters")

        # Instantiate model
        xgb_model = XGBClassifier()
        xgb_model.set_params(**best_params)

        # Fit the train data
        xgb_model.fit(X_train, y_train)

    else:
        logging.info(f"Training xgboost model vanilla")

        # Train xgboost model
        model = train_xgboost_model(
            features_train=X_train, target_train=y_train, weights=sample_weights
        )

# train catboost model
if CHOSEN_MODEL == "catboost":
    logging.info(f"Model chosen for training: catboost")

    # Load the data into pool
    train_pool = Pool(data=X_train, label=y_train)
    test_pool = Pool(data=X_test, label=y_test)

    if cfg["tune_hyperparameters"]:
        logging.info(f"Tuning catboost hyperparameters")

        # Define the search space
        space = {
            "learning_rate": hp.uniform(
                "learning_rate",
                cfg["hyperparameters"]["catboost"]["learning_rate"]["min"],
                cfg["hyperparameters"]["catboost"]["learning_rate"]["max"],
            ),
            "depth": hp.randint("depth", cfg["hyperparameters"]["catboost"]["depth"]),
            "l2_leaf_reg": hp.uniform(
                "l2_leaf_reg",
                cfg["hyperparameters"]["catboost"]["l2_leaf_reg"]["min"],
                cfg["hyperparameters"]["catboost"]["l2_leaf_reg"]["max"],
            ),
            "random_strength": hp.uniform(
                "random_strength",
                cfg["hyperparameters"]["catboost"]["random_strength"]["min"],
                cfg["hyperparameters"]["catboost"]["random_strength"]["max"],
            ),
        }
        # Get the best hyperparameters combination
        best_params = tune_hyerparameters(
            model_type=CHOSEN_MODEL,
            search_space=space,
            features_train=X_train,
            features_test=X_test,
            target_train=y_train,
            target_test=y_test,
        )

        # Instantiate model
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=best_params["learning_rate"],
            random_strength=best_params["random_strength"],
            depth=best_params["depth"],
            loss_function="MultiClass",
            eval_metric="AUC",
            leaf_estimation_method="Newton",
            l2_leaf_reg=best_params["l2_leaf_reg"],
        )

        # Train model with hyperparameters
        model.fit(train_pool, plot=True, eval_set=test_pool)

    # Get the pred on test data
    else:
        logging.info(f"Training catboost model vanilla")

        # Train catboost model
        model = train_catboost_model(
            train_pool == train_pool,
            test_pool=test_pool,
            iterations=cfg["cat_boost_model"]["iterations"],
            learning_rate=cfg["cat_boost_model"]["learning_rate"],
            random_strength=cfg["cat_boost_model"]["random_strength"],
            depth=cfg["cat_boost_model"]["depth"],
            loss_function=cfg["cat_boost_model"]["loss_function"],
            eval_metric=cfg["cat_boost_model"]["eval_metric"],
            leaf_estimation_method=cfg["cat_boost_model"]["leaf_estimation_method"],
            random_seed=cfg["random_seed"],
        )

# Evaluate the performane of trained model
logging.info(f"Evaluating trained model: {CHOSEN_MODEL} performance ...")

# Get the pred on test data
EVAL_RES_PATH = cfg["data"]["final_results_path"]
get_classification_report(
    train_model=model,
    features_test=X_test,
    target_test=y_test,
    save_results_path=EVAL_RES_PATH,
)

# Calculate the roc score
ROC_CURVE_PATH = cfg["data"]["final_results_roc_curve_path"]
auc_score = get_auc_score(
    trained_model=model,
    features_test=X_test,
    target_test=y_test,
    save_roc_plot_path=ROC_CURVE_PATH,
)

logging.info(
    f"Evaluation complete. AUC test score: {auc_score}. Classification report saved to: {EVAL_RES_PATH}. ROC curve saved to: {ROC_CURVE_PATH}"
)


# Save the trained model
SAVE_MODEL_PATH = cfg["model"]["save_model_artifacts_folder"]
save_trained_model(
    model_type=CHOSEN_MODEL, trained_model=model, save_model_path=SAVE_MODEL_PATH
)


if __name__ == "__main__":
    pass
