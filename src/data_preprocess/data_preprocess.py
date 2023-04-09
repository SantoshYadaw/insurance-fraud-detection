import os
import sys
import inspect
import logging

import pandas as pd

# Append path to parent directory (src) for easier import
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from src.utils import load_yaml_config
from src.data_preprocess.data_preprocessing_utils import (
    drop_columns,
    split_data,
    perform_ordinal_encoding,
    perform_nominal_encoding,
    perform_label_encoding,
    perform_numerical_normalization,
)

# Set logging
logging.basicConfig(level=logging.INFO, force=True)

# Loading config file
# cfg = load_yaml_config("../config/config.yaml")
CFG_PATH = os.path.join(parentdir, "config", "config.yaml")
logging.info("CFG_PATH: {CFG_PATH}")
cfg = load_yaml_config(CFG_PATH)

# Load raw data
logging.info(f"Loading the raw data for preprocessing")

insurance_data = pd.read_csv(cfg["data"]["raw_data_path"])

# Drop any redundant / correlated columns
logging.info(f"Dropping any redundant or correlated columns")

# Specify the feature and target col
COLS = list(insurance_data.columns)
TARGET_COL = cfg["data"]["data_preprocess"]["target_column"]
logging.info(f"TARGET_COL: {TARGET_COL}")

DATETIME_COLS = cfg["data"]["data_preprocess"]["datetime_columns"]
logging.info(f"DATETIME_COLS: {DATETIME_COLS}")

REDUNDANT_COLS = cfg["data"]["data_preprocess"]["redundant_columns"]
logging.info(f"REDUNDANT_COLS: {REDUNDANT_COLS}")

NUMERICAL_COLS = cfg["data"]["data_preprocess"]["numerical_columns"]
logging.info(f"NUMERICAL_COLS: {NUMERICAL_COLS}")

# Final set of feature cols
FEATURE_COLS = [
    col
    for col in COLS
    if col not in TARGET_COL
    if col not in DATETIME_COLS
    if col not in REDUNDANT_COLS
    if col not in NUMERICAL_COLS
]

# Split into Train and Test
logging.info(f"Split into Train and Test")

# Split dataset into features and target variable columns
X, y = drop_columns(data=insurance_data, config=cfg)

# Fill the na for those unknown customer education
X["CUSTOMER_EDUCATION_LEVEL"].fillna("unknown", inplace=True)

# Stratified split
X_train, X_test, y_train, y_test = split_data(features=X, target=y, config=cfg)

# Transform categorical columns into numerical columns
logging.info(f"Transform categorical columns into numerical columns")

# Further breakdown of feature cols
ORDINAL_COLS = cfg["data"]["data_preprocess"]["ordinal_columns"]
logging.info(f"ORDINAL_COLS: {ORDINAL_COLS}")

NOMINAL_COLS = [
    col for col in FEATURE_COLS if col not in ORDINAL_COLS if col not in NUMERICAL_COLS
]  # has no order

# Checking
assert len(NOMINAL_COLS) + len(ORDINAL_COLS) == len(FEATURE_COLS)

# Perform ordinal encoding
logging.info("Perform ordinal encoding")
train_data_ordinal, test_data_ordinal = perform_ordinal_encoding(
    ordinal_columns=ORDINAL_COLS, features_train=X_train, features_test=X_test
)

# Perform label encoding
logging.info("Perform label encoding on target variable")
le_class_train, le_class_test = perform_label_encoding(
    target_train=y_train, target_test=y_test
)

# Perform using binary encoder - we use binary encoder as if we use ohe it will result in sparse matrix that affects memory when training
logging.info("Perform binary encoding")
(
    insurance_train_data_nominal_transformer,
    insurance_test_data_nominal_transformer,
) = perform_nominal_encoding(
    nominal_columns=NOMINAL_COLS, features_train=X_train, features_test=X_test
)

# Normalize the numerical columns
logging.info(f"Normalize the numerical columns")
(
    insurance_train_data_numerical,
    insurance_test_data_numerical,
) = perform_numerical_normalization(
    numerical_columns=NUMERICAL_COLS, features_train=X_train, features_test=X_test
)

# Combine and save data
logging.info(f"Combine and save")

X_train = pd.merge(
    pd.merge(
        train_data_ordinal,
        insurance_train_data_nominal_transformer,
        left_index=True,
        right_index=True,
    ),
    insurance_train_data_numerical,
    left_index=True,
    right_index=True,
)
X_test = pd.merge(
    pd.merge(
        test_data_ordinal,
        insurance_test_data_nominal_transformer,
        left_index=True,
        right_index=True,
    ),
    insurance_test_data_numerical,
    left_index=True,
    right_index=True,
)
y_train = le_class_train
y_test = le_class_test

data_train_final = X_train.copy()
data_train_final[cfg["data"]["target_variable"]] = y_train

data_test_final = X_test.copy()
data_test_final[cfg["data"]["target_variable"]] = y_test

PROCESSED_TRAIN_DATA_FILE_PATH = cfg["data"]["train_data_path"]
data_train_final.to_csv(PROCESSED_TRAIN_DATA_FILE_PATH, index=False)
logging.info(f"Processed train data saved to: {PROCESSED_TRAIN_DATA_FILE_PATH}")

PROCESSED_TEST_DATA_FILE_PATH = cfg["data"]["test_data_path"]
data_test_final.to_csv(PROCESSED_TEST_DATA_FILE_PATH, index=False)
logging.info(f"Processed test data saved to: {PROCESSED_TEST_DATA_FILE_PATH}")


if __name__ == "__main__":
    pass
