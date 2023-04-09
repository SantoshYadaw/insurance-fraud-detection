import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OrdinalEncoder,
    LabelEncoder,
    StandardScaler,
)
import category_encoders as ce

# from utils import load_yaml_config

# Set logging
logging.basicConfig(level=logging.INFO, force=True)


# Helper function 1: drop columns
def drop_columns(data: pd.DataFrame, config: dict):
    """Drop unwanted columns from dataframe

    Args:
        data (pd.DataFrame): Input data
        config (dict): Configuration file

    Returns:
        pd.DatFrame: Features and Target
    """
    # Drop any redundant / correlated columns
    logging.info(f"Dropping any redundant or correlated columns")

    # Specify the feature and target col
    COLS = list(data.columns)
    TARGET_COL = config["data"]["data_preprocess"]["target_column"]
    logging.info(f"TARGET_COL: {TARGET_COL}")

    DATETIME_COLS = config["data"]["data_preprocess"]["datetime_columns"]
    logging.info(f"DATETIME_COLS: {DATETIME_COLS}")

    REDUNDANT_COLS = config["data"]["data_preprocess"]["redundant_columns"]
    logging.info(f"REDUNDANT_COLS: {REDUNDANT_COLS}")

    NUMERICAL_COLS = config["data"]["data_preprocess"]["numerical_columns"]
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
    features = data.drop(TARGET_COL, axis=1)
    features = data.drop(REDUNDANT_COLS, axis=1)
    target = data[TARGET_COL]

    return features, target


def split_data(features: pd.DataFrame, target: pd.DataFrame, config: dict):
    """Split the data into training and test set

    Args:
        features (pd.DataFrame): Features data
        target (pd.DataFrame): Target data
        config (dict): Configuration

    Returns:
        pd.DataFrame: Data split according to train and test by features and target
    """
    if config["data"]["data_preprocess"]["train_test_split"]["to_stratify"]:
        logging.info(f"Stratify split method selected")

        # Stratified split
        features_train, features_test, target_train, target_test = train_test_split(
            features,
            target,
            test_size=config["data"]["data_preprocess"]["train_test_split"][
                "split_ratio"
            ],
            stratify=target,
            random_state=config["random_seed"],
            shuffle=True,
        )
    else:
        logging.info(f"Random split method selected")

        features_train, features_test, target_train, target_test = train_test_split(
            features,
            target,
            test_size=config["data"]["data_preprocess"]["train_test_split"][
                "split_ratio"
            ],
            stratify=target,
            random_state=config["random_seed"],
            shuffle=True,
        )

    return features_train, features_test, target_train, target_test


def perform_ordinal_encoding(
    ordinal_columns: list, features_train: pd.DataFrame, features_test: pd.DataFrame
):
    """Peform encoding on ordinal features - there is order to them

    Args:
        ordinal_columns (list): List of ordinal columns
        features_train (pd.DataFrame): Features from train split
        features_test (pd.DataFrame): Features from test split

    Returns:
        pd.DataFrame: Features from train and test after ordinal encoding transformation
    """
    # Instantiate ordincal encoder
    ord_enc = OrdinalEncoder()

    # Fit and transform the data
    features_ordinal_train = ord_enc.fit_transform(features_train[ordinal_columns])
    featues_ordinal_test = ord_enc.transform(features_test[ordinal_columns])

    tranformed_features_ordinal_train = pd.DataFrame(
        features_ordinal_train,
        columns=ordinal_columns,
        index=features_train[ordinal_columns].index,
    )
    tranformed_features_ordinal_test = pd.DataFrame(
        featues_ordinal_test,
        columns=ordinal_columns,
        index=features_test[ordinal_columns].index,
    )

    return tranformed_features_ordinal_train, tranformed_features_ordinal_test


def perform_nominal_encoding(
    nominal_columns: list, features_train: pd.DataFrame, features_test: pd.DataFrame
):
    """Pefrom encoding on nominal features - there is no order

    Args:
        nominal_columns (list): List of nominal features
        features_train (pd.DataFrame): Features from train split
        features_test (pd.DataFrame): Features from test split

    Returns:
        pd.DataFrame: Features from train and test after nominal encoding transformation
    """
    # Instantiate binary encoder
    be_enc = ce.BinaryEncoder(cols=nominal_columns)

    features_nominal_train = be_enc.fit_transform(features_train[nominal_columns])
    features_nominal_test = be_enc.transform(features_test[nominal_columns])

    return features_nominal_train, features_nominal_test


def perform_label_encoding(target_train: pd.DataFrame, target_test: pd.DataFrame):
    """Perform label encoding on target variable both train and test split

    Args:
        target_train (pd.DataFrame): Target variable train split
        target_test (pd.DataFrame): Target variable test split

    Returns:
        pd.DataFrame: Target variable from train and test after label encoding transformation
    """
    # Instantiate label encoder for target variable
    le = LabelEncoder()

    # Transform
    transformed_target_train = le.fit_transform(target_train)
    transformed_target_test = le.transform(target_test)

    return transformed_target_train, transformed_target_test


def perform_numerical_normalization(
    numerical_columns: list, features_train: pd.DataFrame, features_test: pd.DataFrame
):
    """Perform numerical normalization

    Args:
        numerical_columns (list): Numerical columns
        features_train (pd.DataFrame): Features from train split
        features_test (pd.DataFrame): Features from test split

    Returns:
        pd.DatFrame: Features from train and test after numerical normalization
    """
    # instantiate scaler
    scaler = StandardScaler()

    # Fit and transform the data
    feaures_numerical_train = scaler.fit_transform(features_train[numerical_columns])
    feaures_numerical_test = scaler.transform(features_test[numerical_columns])

    feaures_numerical_train_transformed = pd.DataFrame(
        feaures_numerical_train,
        columns=numerical_columns,
        index=features_train[numerical_columns].index,
    )
    feaures_numerical_test_transformed = pd.DataFrame(
        feaures_numerical_test,
        columns=numerical_columns,
        index=features_test[numerical_columns].index,
    )

    return feaures_numerical_train_transformed, feaures_numerical_test_transformed


if __name__ == "__main__":
    pass
