import os
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OrdinalEncoder,
    LabelEncoder,
    StandardScaler,
)
import category_encoders as ce

from utils import load_yaml_config

# Set logging
logging.basicConfig(level=logging.INFO, force=True)

# Loading config file
logging.info(f"Loading the config file")
cfg = load_yaml_config("../config/config.yaml")

# Load raw data
logging.info(f"Loading the raw data for preprocessing")

insurance_data = pd.read_csv(cfg["data"]["raw_data_path"])

# Drop any redundant / correlated columns
logging.info(f"Dropping any redundant or correlated columns")

# Specify the feature and target col
COLS = list(insurance_data.columns)
TARGET_COL = ["CLAIM_STATUS"]
DATETIME_COLS = ["TXN_DATE_TIME", "POLICY_EFF_DT", "LOSS_DT", "REPORT_DT"]

# Able to retrieve these information from the postal code column
REDUNDANT_COLS = [
    "ADDRESS_LINE1",
    "ADDRESS_LINE2",
    "CITY",
    "STATE",
    "POLICY_NUMBER",
    "TRANSACTION_ID",
    "VENDOR_ID",
    "TXN_DATE_TIME",
    "POLICY_EFF_DT",
    "LOSS_DT",
    "REPORT_DT",
    "SSN",
]

# Numerical cols
NUMERICAL_COLS = [
    "PREMIUM_AMOUNT",
    "POSTAL_CODE",
    "NO_OF_FAMILY_MEMBERS",
    "ROUTING_NUMBER",
    "ANY_INJURY",
    "POLICE_REPORT_AVAILABLE",
    "INCIDENT_HOUR_OF_THE_DAY",
    "CLAIM_AMOUNT",
    "AGE",
    "TENURE",
]

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
X = insurance_data.drop(TARGET_COL, axis=1)
X = insurance_data.drop(REDUNDANT_COLS, axis=1)
y = insurance_data[TARGET_COL]
X["CUSTOMER_EDUCATION_LEVEL"].fillna("unknown", inplace=True)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=cfg["data"]["train_test_split"],
    stratify=y,
    random_state=cfg["random_seed"],
    shuffle=True,
)

# Transform categorical columns into numerical columns
logging.info(f"Transform categorical columns into numerical columns")

# Further breakdown of feature cols
ORDINAL_COLS = [
    "RISK_SEGMENTATION",
    "SOCIAL_CLASS",
    "CUSTOMER_EDUCATION_LEVEL",
    "INCIDENT_SEVERITY",
]  # has some order
NOMINAL_COLS = [
    col for col in FEATURE_COLS if col not in ORDINAL_COLS if col not in NUMERICAL_COLS
]  # has no order

# Checking
assert len(NOMINAL_COLS) + len(ORDINAL_COLS) == len(FEATURE_COLS)

# Instantiate ordincal encoder
ord_enc = OrdinalEncoder()

# Fit and transform the data
X_ord_enc_train = ord_enc.fit_transform(X_train[ORDINAL_COLS])
X_ord_enc_test = ord_enc.transform(X_test[ORDINAL_COLS])

train_data_ordinal = pd.DataFrame(
    X_ord_enc_train, columns=ORDINAL_COLS, index=X_train[ORDINAL_COLS].index
)
test_data_ordinal = pd.DataFrame(
    X_ord_enc_test, columns=ORDINAL_COLS, index=X_test[ORDINAL_COLS].index
)

# Instantiate label encoder for target variable
le = LabelEncoder()
le_class_train = le.fit_transform(y_train)
le_class_test = le.transform(y_test)

# Perform using binary encoder - we use binary encoder as if we use ohe it will result in sparse matrix that affects memory when training

# Instantiate ordincal encoder
be_enc = ce.BinaryEncoder(cols=NOMINAL_COLS)

# Fit and transform the data
insurance_train_data_nominal_transformer = be_enc.fit_transform(X_train[NOMINAL_COLS])
insurance_test_data_nominal_transformer = be_enc.transform(X_test[NOMINAL_COLS])

# Normalize the numerical columns
logging.info(f"Normalize the numerical columns")

# instantiate scaler
scaler = StandardScaler()

# Fit and transform the data
X_numerical_train = scaler.fit_transform(X_train[NUMERICAL_COLS])
X_test_train = scaler.transform(X_test[NUMERICAL_COLS])

insurance_train_data_numerical = pd.DataFrame(
    X_numerical_train, columns=NUMERICAL_COLS, index=X_train[NUMERICAL_COLS].index
)
insurance_test_data_numerical = pd.DataFrame(
    X_test_train, columns=NUMERICAL_COLS, index=X_test[NUMERICAL_COLS].index
)

# 6. Combine and save
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
