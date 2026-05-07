"""
Where possible we preprocessed the datasets as done by Athavale et al. (2024) in CAV. 
"""


import os
import pandas as pd
from .encoding import (
    produce_categorical_map,
    produce_min_max_map,
    encode_row,
    build_categorical_numeric_info,
)
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as pkl
import kagglehub
import torch
from torch.utils.data import DataLoader, TensorDataset

# for the german credit dataset
import requests
import zipfile
import io

from typing import NamedTuple, Any


def split_and_encode_dataset(
    X,
    y,
    numeric_cols,
    categorical_cols,
    seed=42,
    test_size=0.2,
    val_size=0.2,
    encoding_size=20,
    cat_pickle_path=None,
    num_pickle_path=None,
    onlyFloat=False,
):
    """
    Splits data into training, validation, and test sets and then encodes each row.

    Parameters:
        X (pd.DataFrame): The feature DataFrame.
        y (pd.Series or np.array): The target.
        numeric_cols (list): List of numeric column names.
        categorical_cols (list): List of categorical column names.
        seed (int): Random seed for splits.
        test_size (float): Fraction of data for the test set.
        val_size (float): Fraction of the remaining data for the validation set.
        encoding_size (int): Maximum encoding size (for categorical variables).
        cat_pickle_path (str): If provided, path to save the categorical info pickle.
        num_pickle_path (str): If provided, path to save the numeric info pickle.

    Returns:
        X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test,
        cat_info, num_info
    """
    # First split: training+validation vs. test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Second split: training vs. validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_val,
    )

    cat_unique_map = produce_categorical_map(X, categorical_cols)
    min_max_map = produce_min_max_map(X_train, numeric_cols)
    columns = X.columns if hasattr(X, "columns") else None
    assert columns is not None, "X must have a 'columns' attribute."

    cat_info, num_info, total_dim = build_categorical_numeric_info(
        cat_unique_map,
        min_max_map,
        max_encoding_size=encoding_size,
        columns=columns,
        onlyFloat=onlyFloat,
    )

    # Optionally save the mappings to pickle files
    if cat_pickle_path is not None:
        with open(cat_pickle_path, "wb") as handle:
            pkl.dump(cat_info, handle, protocol=pkl.HIGHEST_PROTOCOL)
    if num_pickle_path is not None:
        with open(num_pickle_path, "wb") as handle:
            pkl.dump(num_info, handle, protocol=pkl.HIGHEST_PROTOCOL)

    # Define a helper to encode each row
    encode_func = lambda r: encode_row(
        r,
        numeric_cols,
        cat_unique_map,
        min_max_map,
        cat_info,
        num_info,
        total_dim,
        max_encoding_size=encoding_size,
        columns=columns,
        onlyFloat=onlyFloat,
    )
    #exit()
    # Apply encoding to train, validation, and test sets
    train_encoded_series = X_train.apply(encode_func, axis=1)
    val_encoded_series = X_val.apply(encode_func, axis=1)
    test_encoded_series = X_test.apply(encode_func, axis=1)

    # Convert series of encoded rows to numpy arrays
    X_train_encoded = np.array([list(row) for row in train_encoded_series.values])
    X_val_encoded = np.array([list(row) for row in val_encoded_series.values])
    X_test_encoded = np.array([list(row) for row in test_encoded_series.values])
    return (
        X_train_encoded,
        y_train,
        X_val_encoded,
        y_val,
        X_test_encoded,
        y_test,
        cat_info,
        num_info,
    )


def build_law_ds(seed=42, info_path="models"):
    # kaggle_path = '/nfs/scistore16/tomgrp/fkresse/.cache/kagglehub/datasets'
    kaggle_path = os.environ.get(
        "KAGGLEHUB_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".kagglehub", "datasets"),
    )
    kaggle_path_law = (
        kaggle_path
        + "/danofer/law-school-admissions-bar-passage/versions/1/bar_pass_prediction.csv"
    )
    print(kaggle_path_law)
    if not os.path.exists(kaggle_path_law):
        print("Downloading dataset")
        path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
        path = path + "/bar_pass_prediction.csv"
    else:
        print("Path exists")
        path = kaggle_path_law
    data = pd.read_csv(path)
    # see below for why which features dropped
    # https://www.kaggle.com/datasets/danofer/law-school-admissions-bar-passage/discussion/350765
    data = data.drop(
        columns=[
            "decile1b",
            "decile3",
            "ID",
            "decile1",
            "cluster",
            "gpa",
            "male",
            "zgpa",
            "zfygpa",
            "bar1",
            "bar1_yr",
            "bar2",
            "bar2_yr",
            "bar_passed",
            "race",
            "race2",
            "asian",
            "hisp",
            "other",
            "black",
            "Dropout",
            "parttime",
            "DOB_yr",
            "age",
            "index6040",
            "indxgrp",
            "indxgrp2",
            "dnn_bar_pass_prediction",
            "bar",
            "sex",
        ]
    )
    data = data.dropna()
    # rename race1 to race
    data = data.rename(columns={"race1": "race"})
    data["ugpa"] = (data["ugpa"] * 10).round().astype(int)
    data_X = data.drop(columns=["pass_bar"])
    data_y = data["pass_bar"].values
    numeric_cols = ["lsat", "ugpa", "fam_inc", "tier"]
    categorical_cols = [col for col in data_X.columns if col not in numeric_cols]
    return split_and_encode_dataset(
        data_X,
        data_y,
        numeric_cols,
        categorical_cols,
        seed=seed,
        encoding_size=20,
        cat_pickle_path=f"{info_path}/categorical_law_{seed}.pickle",
        num_pickle_path=f"{info_path}/num_info_law_{seed}.pickle",
    )


def build_compas_ds(seed=42, info_path="models"):

    # kaggle_path = '/nfs/scistore16/tomgrp/fkresse/.cache/kagglehub/datasets'
    kaggle_path = os.environ.get(
        "KAGGLEHUB_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".kagglehub", "datasets"),
    )
    kaggle_path_compass = (
        kaggle_path + "/danofer/compass/versions/1/compas-scores-raw.csv"
    )
    print(kaggle_path_compass)
    if not os.path.exists(kaggle_path_compass):
        print("Downloading dataset")
        path = kagglehub.dataset_download("danofer/compass")
        path = path + "/compas-scores-raw.csv"
    else:
        print("Path exists")
        path = kaggle_path_compass
    data = pd.read_csv(path)

    # path = kagglehub.dataset_download("danofer/compass")
    df = data.copy()
    columns_to_keep = df.columns[[7, 8, 11, 14, 18, 21, 24]]
    dropped = df[columns_to_keep]
    dropped = dropped.rename(
        columns={"Sex_Code_Text": "gender", "Ethnic_Code_Text": "race"}
    )

    dropped = dropped.dropna()

    data_features = dropped.iloc[:, :-1]
    print(data_features.columns)
    print(data_features.iloc[0])
    y = dropped.iloc[:, -1]
    y = y.map({"Low": 0, "Medium": 1, "High": 2}).values

    numeric_cols = ["RecSupervisionLevel"]
    categorical_cols = [col for col in data_features.columns if col not in numeric_cols]
    # print(categorical_cols)
    return split_and_encode_dataset(
        data_features,
        y,
        numeric_cols,
        categorical_cols,
        seed=seed,
        encoding_size=20,
        cat_pickle_path=f"{info_path}/categorical_compas_{seed}.pickle",
        num_pickle_path=f"{info_path}/num_info_compas_{seed}.pickle",
    )


def build_adult_ds(seed=42, info_path="models"):

    kaggle_path = os.environ.get(
        "KAGGLEHUB_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".kagglehub", "datasets"),
    )
    kaggle_path_adult = (
        kaggle_path + "/wenruliu/adult-income-dataset/versions/2/adult.csv"
    )

    if not os.path.exists(kaggle_path_adult):
        print("Downloading dataset")
        path = kagglehub.dataset_download("wenruliu/adult-income-dataset")
        path = path + "/adult.csv"
    else:
        print("Path exists")
        path = kaggle_path_adult
    data = pd.read_csv(path)
    # inspired by https://cseweb.ucsd.edu//classes/sp15/cse190-c/reports/sp15/048.pdf
    data = data.drop(
        columns=[
            "fnlwgt",
            "education",
            "capital-gain",
            "capital-loss",
            "relationship",
            "native-country",
        ]
    )

    data = data.replace("?", pd.NA).dropna()
    data_X = data.iloc[:, :-1]
    data_y = data.iloc[:, -1].map({"<=50K": 0, ">50K": 1}).values
    numeric_cols = ["age", "educational-num", "hours-per-week"]
    categorical_cols = [col for col in data_X.columns if col not in numeric_cols]

    return split_and_encode_dataset(
        data_X,
        data_y,
        numeric_cols,
        categorical_cols,
        seed=seed,
        encoding_size=20,
        cat_pickle_path=f"{info_path}/categorical_adult_{seed}.pickle",
        num_pickle_path=f"{info_path}/num_info_adult_{seed}.pickle",
    )


def build_mnist(seed=42, info_path="models"):
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1)
    X, y = mnist["data"], mnist["target"]
    X = X / 255.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_folktable_5(seed=42, info_path="models"):
    # inspired by https://cseweb.ucsd.edu//classes/sp15/cse190-c/reports/sp15/048.pdf
    from folktables import ACSDataSource, ACSIncome, adult_filter, BasicProblem
    import os

    kaggle_path = os.environ.get(
        "KAGGLEHUB_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".kagglehub", "datasets"),
    )
    # check if ca_features exists
    if os.path.exists("ca_features.csv") and os.path.exists("ca_labels.csv"):
        ca_features = pd.read_csv("ca_features.csv")
        ca_labels = pd.read_csv("ca_labels.csv")
    else:
        
        data_source = ACSDataSource(
            survey_year="2018", horizon="1-Year", survey="person"
        )
        ca_data = data_source.get_data(states=["CA"], download=True)

        ACSIncome = BasicProblem(
            features=[
                "AGEP",
                "COW",
                "SCHL",
                "MAR",
                # 'OCCP',
                # 'POBP',
                # 'RELP',
                "WKHP",
                "SEX",
                "RAC1P",
            ],
            target="PINCP",
            target_transform=lambda x: np.select(
                [x <= 20000, x <= 40000, x <= 60000, x <= 80000, x > 80000],
                [0, 1, 2, 3, 4],
                default=-1,  # Fallback value (optional)
            ),  # target transform into 5 buckets
            group="RAC1P",
            preprocess=adult_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        ca_features, ca_labels, _ = ACSIncome.df_to_pandas(ca_data)
        #print(np.unique(ca_labels.values, return_counts=True))
        ca_features.to_csv("ca_features.csv", index=False)
        ca_labels.to_csv("ca_labels.csv", index=False)
    # in each column print unique number
    #for col in ca_features.columns:
    #    print(col, len(np.unique(ca_features[col].values)))
    # print(data[0].columns)
    # rename columns SEX and RAC1P to gender and race
    ca_features = ca_features.rename(
        columns={"SEX": "gender", "RAC1P": "race", "AGEP": "age"}
    )
    numeric_cols = ["age", "SCHL", "WKHP"]
    categorical_cols = [col for col in ca_features.columns if col not in numeric_cols]
    y = ca_labels.values.squeeze()
    # print(ca_features.shape)
    return split_and_encode_dataset(
        ca_features,
        y,
        numeric_cols,
        categorical_cols,
        seed=seed,
        encoding_size=20,
        cat_pickle_path=f"{info_path}/categorical_folktables5_{seed}.pickle",
        num_pickle_path=f"{info_path}/num_info_folktables5_{seed}.pickle",
    )


def download_german_credit(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)  # Create the directory if it doesn't exist
    # URL of the dataset
    url = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
    # Download the dataset
    response = requests.get(url)
    if response.status_code == 200:
        # Open the downloaded content as a zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract all contents into the cache directory
            z.extractall(cache_dir)
        print("Dataset downloaded and extracted to:", cache_dir)
    else:
        print("Failed to download dataset. Status code:", response.status_code)
        raise ValueError("Failed to download dataset german credit.")
    # return the path
    return cache_dir


def build_german_credit_ds(seed=42, info_path="models"):

    # For the German Credit dataset, read from file and do some remapping.
    # data = pd.read_csv('german.data')
    # check if the .data exists
    cache_dir = os.environ.get(
        "KAGGLEHUB_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".kagglehub", "datasets"),
    )
    if not os.path.exists(cache_dir + "/german.data"):
        path = download_german_credit(cache_dir)
    else:
        path = cache_dir

    df = pd.read_csv(path + "/german.data", delimiter="\\s+", header=None)
    # TODO! make german data downloadedable if not there!

    # Assume that numeric columns are known (by index or name)
    # column 9 is guarantors, dependents is 17, phones is 18, 15 (existing credits), 10 (present residence since), maybe 4 credit account
    df = df.drop([18, 10, 15, 17])
    # print(df)
    # exit()
    numeric_cols = [1, 4, 7, 10, "age", 15, 17]
    numeric_cols = [1, 4, 7, "age"]
    # Take all columns except the last as features.
    data_features = df.iloc[:, :-1].copy()

    # Rename column 8 (personal status/sex) to 'gender', column 12 to 'age'
    data_features = data_features.rename(columns={8: "gender", 12: "age"})

    # Map the gender codes to 'male'/'female'
    data_features["gender"] = data_features["gender"].map(
        {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}
    )

    categorical_cols = [col for col in data_features.columns if col not in numeric_cols]

    # The target is in the last column; adjust values if needed.
    y = df.iloc[:, -1].values.astype(int) - 1

    return split_and_encode_dataset(
        data_features,
        y,
        numeric_cols,
        categorical_cols,
        seed=seed,
        encoding_size=5,
        cat_pickle_path=f"{info_path}/categorical_german_credit_{seed}.pickle",
        num_pickle_path=f"{info_path}/num_info_german_credit_{seed}.pickle",
    )


import os
import numpy as np
import pandas as pd
from folktables import ACSDataSource


def detect_column_types(features, forced_categorical=None, unique_threshold=None):
    """
    Determines which columns are numeric vs categorical.

    Args:
        features: DataFrame of features.
        forced_categorical: List of columns that should always be categorical.
        unique_threshold: If provided, any column with fewer unique values than
                          this threshold will be treated as categorical.

    Returns:
        numeric_cols: List of columns to treat as numeric.
        categorical_cols: List of columns to treat as categorical.
    """
    forced_categorical = forced_categorical or []
    numeric_cols = []
    categorical_cols = []

    for col in features.columns:
        if col in forced_categorical:
            categorical_cols.append(col)
        else:
            # If unique_threshold is provided and the number of unique values is below it, mark as categorical.
            if (
                unique_threshold is not None
                and features[col].nunique() < unique_threshold
            ):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
    return numeric_cols, categorical_cols


import os
import numpy as np
import pandas as pd
from folktables import ACSDataSource, generate_categories
import os
import numpy as np
import pandas as pd
from folktables import ACSDataSource, generate_categories


def build_folktable_task(
    task,
    seed=42,
    info_path="models",
    states=["CA"],
    survey_year="2018",
    horizon="1-Year",
    survey="person",
    encoding_size=20,
    save_path=None,
    recreate=False,
):
    """
    Build a prediction task using a Folktables task instance while preserving the
    original dataframe. The categorical columns are automatically determined from
    the categories dictionary generated via ACS definitions. This function:

      1. Downloads the ACS data for the specified states.
      2. Downloads ACS definitions and generates a categories dictionary using the
         task's feature list.
      3. Converts the raw data to a pandas DataFrame (without dummy encoding).
      4. Determines categorical columns as those whose names appear as keys in the
         categories dictionary, and numeric columns as all remaining columns.
      5. Calls split_and_encode_dataset on the resulting data.

    Args:
        task: A Folktables prediction task (e.g., ACSIncome, ACSEmployment, etc.)
        seed: Random seed for splitting/encoding.
        info_path: Directory where pickle files for encoded info are saved.
        states: List of states to download data from.
        survey_year: ACS survey year.
        horizon: ACS horizon (e.g., '1-Year').
        survey: ACS survey type (e.g., 'person').
        encoding_size: Maximum encoding size for numeric features.

    Returns:
        The output of split_and_encode_dataset on the processed features and labels.
    """

    if save_path is not None:
        # check if exists else create the folders
        os.makedirs(save_path, exist_ok=True)
        required_files = [
            "X_train.npy",
            "y_train.npy",
            "X_valid.npy",
            "y_valid.npy",
            "X_test.npy",
            "y_test.npy",
        ]
        save_files_exist = all(
            os.path.exists(os.path.join(save_path, f)) for f in required_files
        )
        if save_files_exist and not recreate:
            print("Loading dataset splits from", save_path)
            X_train = np.load(os.path.join(save_path, "X_train.npy"), allow_pickle=True)
            y_train = np.load(os.path.join(save_path, "y_train.npy"), allow_pickle=True)
            X_valid = np.load(os.path.join(save_path, "X_valid.npy"), allow_pickle=True)
            y_valid = np.load(os.path.join(save_path, "y_valid.npy"), allow_pickle=True)
            X_test = np.load(os.path.join(save_path, "X_test.npy"), allow_pickle=True)
            y_test = np.load(os.path.join(save_path, "y_test.npy"), allow_pickle=True)
            # load the encodings
            with open(os.path.join(save_path, "enc_n.pickle"), "rb") as handle:
                enc_n = pkl.load(handle)
            with open(os.path.join(save_path, "enc_d.pickle"), "rb") as handle:
                enc_d = pkl.load(handle)
            return X_train, y_train, X_valid, y_valid, X_test, y_test, enc_n, enc_d

    # Ensure info_path directory exists
    os.makedirs(info_path, exist_ok=True)

    # Instantiate the ACS data source
    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey)

    # Download raw data for the specified states
    data = data_source.get_data(states=states, download=True)

    # Download ACS definitions and automatically generate the categories dictionary
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(
        features=task.features, definition_df=definition_df
    )

    # Convert the raw data into a pandas DataFrame without dummy encoding.
    features, labels, _ = task.df_to_pandas(data, categories=categories, dummies=False)

    # Determine which columns are categorical based on the keys in the categories dictionary.
    categorical_cols = [col for col in features.columns if col in categories]
    numeric_cols = [col for col in features.columns if col not in categorical_cols]
    #print(categorical_cols)
    #print(numeric_cols)
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    # Ensure labels are a 1D array
    y = labels.values.squeeze()

    # Generate file names for storing encoding information
    task_name = task.__class__.__name__
    cat_pickle_path = os.path.join(
        info_path, f"categorical_folktables_{task_name}_{seed}.pickle"
    )
    num_pickle_path = os.path.join(
        info_path, f"num_info_folktables_{task_name}_{seed}.pickle"
    )
    # take only the first 1000 of the features and y
    # features = features.iloc[:1000]
    # y = y[:1000]

    # Call the helper function to split and encode the dataset.
    print("start splits")
    splits = split_and_encode_dataset(
        features,
        y,
        numeric_cols,
        categorical_cols,
        seed=seed,
        encoding_size=encoding_size,
        cat_pickle_path=cat_pickle_path,
        num_pickle_path=num_pickle_path,
    )
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        X_train, y_train, X_valid, y_valid, X_test, y_test, enc_n, enc_d = splits
        np.save(os.path.join(save_path, "X_train.npy"), X_train)
        np.save(os.path.join(save_path, "y_train.npy"), y_train)
        np.save(os.path.join(save_path, "X_valid.npy"), X_valid)
        np.save(os.path.join(save_path, "y_valid.npy"), y_valid)
        np.save(os.path.join(save_path, "X_test.npy"), X_test)
        np.save(os.path.join(save_path, "y_test.npy"), y_test)
        # pickle save the encodings
        with open(os.path.join(save_path, "enc_n.pickle"), "wb") as handle:
            pkl.dump(enc_n, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(save_path, "enc_d.pickle"), "wb") as handle:
            pkl.dump(enc_d, handle, protocol=pkl.HIGHEST_PROTOCOL)

        print("Saved dataset splits to", save_path)
    return splits


def build_folktable_task_old(
    task,
    seed=42,
    info_path="models",
    states=["CA"],
    survey_year="2018",
    horizon="1-Year",
    survey="person",
    encoding_size=20,
    unique_threshold=None,
):
    """
    Build a prediction task using a Folktables task instance.

    Args:
        task: A Folktables prediction task (e.g., ACSIncome, ACSEmployment, etc.)
        seed: Random seed for splitting/encoding.
        info_path: Directory where pickle files for encoded info are saved.
        states: List of states to download data from.
        survey_year: ACS survey year.
        horizon: ACS horizon (e.g., '1-Year').
        survey: ACS survey type (e.g., 'person').
        encoding_size: Maximum encoding size for numeric features.
        unique_threshold: If provided, any column with fewer unique values than this number is
                          treated as categorical.

    Returns:
        The output of split_and_encode_dataset on the processed features and labels.
    """
    # Ensure info_path directory exists
    os.makedirs(info_path, exist_ok=True)

    # Instantiate the data source (modify as needed for non-ACS tasks)
    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey)

    # Get the raw data for the specified states
    data = data_source.get_data(states=states, download=True)

    # Convert data to pandas DataFrame(s) using the provided task helper
    features, labels, _ = task.df_to_pandas(data)

    # Hardcode known categorical columns
    forced_categorical = ["COW", "SCHL", "MAR", "SEX", "RAC1P"]

    # Determine numeric vs. categorical columns
    numeric_cols, categorical_cols = detect_column_types(
        features,
        forced_categorical=forced_categorical,
        unique_threshold=unique_threshold,
    )

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    # Ensure labels are a 1D array.
    y = labels.values.squeeze()

    # Generate file names that depend on the task name and seed.
    task_name = task.__class__.__name__
    cat_pickle_path = os.path.join(
        info_path, f"categorical_folktables_{task_name}_{seed}.pickle"
    )
    num_pickle_path = os.path.join(
        info_path, f"num_info_folktables_{task_name}_{seed}.pickle"
    )

    # Call your helper function to split and encode the dataset.
    return split_and_encode_dataset(
        features,
        y,
        numeric_cols,
        categorical_cols,
        seed=seed,
        encoding_size=encoding_size,
        cat_pickle_path=cat_pickle_path,
        num_pickle_path=num_pickle_path,
    )


def build_iee_cis_fraud(seed=42, info_path="models"):
    kaggle_path = os.environ.get(
        "KAGGLEHUB_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".kagglehub", "datasets"),
    )
    kaggle_path_ieee = os.path.join(kaggle_path, "ieee-fraud-detection")
    print(kaggle_path_ieee)
    if not os.path.exists(kaggle_path_ieee):
        print("Downloading IEEE Fraud Detection dataset")

        # Use the kaggle CLI to download the competition files
        os.system(
            "kaggle competitions download -c ieee-fraud-detection -p {}".format(
                kaggle_path_ieee
            )
        )

        # Unzip the downloaded files
        zip_path = os.path.join(kaggle_path_ieee, "ieee-fraud-detection.zip")
        os.system(f"unzip -o {zip_path} -d {kaggle_path_ieee}")
        path = kaggle_path_ieee
    else:
        print("Path exists")
        path = kaggle_path_ieee
    # check if the numpy files already exist
    if not (os.path.exists(f"{path}/X_train_encoded.npy")):
        print("have to process, will take some time")
        DTYPE = {
            "TransactionID": "int32",
            "isFraud": "int8",
            "TransactionDT": "int32",
            "TransactionAmt": "float32",
            "ProductCD": "category",
            "card1": "int16",
            "card2": "float32",
            "card3": "float32",
            "card4": "category",
            "card5": "float32",
            "card6": "category",
            "addr1": "float32",
            "addr2": "float32",
            "dist1": "float32",
            "dist2": "float32",
            "P_emaildomain": "category",
            "R_emaildomain": "category",
        }

        IDX = "TransactionID"
        TGT = "isFraud"

        CCOLS = [f"C{i}" for i in range(1, 15)]
        DCOLS = [f"D{i}" for i in range(1, 16)]
        MCOLS = [f"M{i}" for i in range(1, 10)]
        VCOLS = [f"V{i}" for i in range(1, 340)]

        DTYPE.update((c, "float32") for c in CCOLS)
        DTYPE.update((c, "float32") for c in DCOLS)
        DTYPE.update((c, "float32") for c in VCOLS)
        DTYPE.update((c, "category") for c in MCOLS)

        DTYPE_ID = {
            "TransactionID": "int32",
            "DeviceType": "category",
            "DeviceInfo": "category",
        }

        ID_COLS = [f"id_{i:02d}" for i in range(1, 39)]
        ID_CATS = [
            "id_12",
            "id_15",
            "id_16",
            "id_23",
            "id_27",
            "id_28",
            "id_29",
            "id_30",
            "id_31",
            "id_33",
            "id_34",
            "id_35",
            "id_36",
            "id_37",
            "id_38",
        ]

        DTYPE_ID.update(((c, "float32") for c in ID_COLS))
        DTYPE_ID.update(((c, "category") for c in ID_CATS))

        IN_DIR = path

        NR = None
        NTRAIN = 590540

        def read_both(t):
            df = pd.read_csv(
                f"{IN_DIR}/{t}_transaction.csv", index_col=IDX, nrows=NR, dtype=DTYPE
            )
            df = df.join(
                pd.read_csv(
                    f"{IN_DIR}/{t}_identity.csv",
                    index_col=IDX,
                    nrows=NR,
                    dtype=DTYPE_ID,
                )
            )
            print(t, df.shape)
            return df

        def read_dataset():
            train = read_both("train")
            # test = read_both('test')
            return train, None

        train, _ = read_dataset()
        prediction = train.pop("isFraud")
        train = train.drop(columns=["DeviceInfo"])
        numeric_columns = []
        categorical_columns = []
        for col in train.columns:
            # print(col, train[col].dtype)
            if train[col].dtype == "category":
                categorical_columns.append(col)
            else:
                numeric_columns.append(col)
        # convert all categorical columns to strings
        for col in categorical_columns:
            train[col] = train[col].astype(str)
        for col in train.columns:
            # if type is object set to category
            if train[col].dtype == "object":
                train[col] = train[col].astype("category")
        train_10 = train  # .head(5000)
        prediction_10 = prediction  # .head(5000)
        print("Starting split and encode")
        out = split_and_encode_dataset(
            train_10,
            prediction_10,
            numeric_columns,
            categorical_columns,
            seed=0,
            encoding_size=5,
            cat_pickle_path=f"categorical_iee_{0}.pickle",
            num_pickle_path=f"num_info_ieee_{0}.pickle",
            onlyFloat=True,
        )
        (
            X_train_encoded,
            y_train,
            X_val_encoded,
            y_val,
            X_test_encoded,
            y_test,
            cat_info,
            num_info,
        ) = out

        # np save all of them to the path
        np.save(f"{path}/X_train_encoded.npy", X_train_encoded)
        np.save(f"{path}/y_train.npy", y_train)
        np.save(f"{path}/X_val_encoded.npy", X_val_encoded)
        np.save(f"{path}/y_val.npy", y_val)
        np.save(f"{path}/X_test_encoded.npy", X_test_encoded)
        np.save(f"{path}/y_test.npy", y_test)
        with open(f"{path}/cat_info.pickle", "wb") as handle:
            pkl.dump(cat_info, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open(f"{path}/num_info.pickle", "wb") as handle:
            pkl.dump(num_info, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print("Finished and saved all the files")
    else:
        X_train_encoded = np.load(f"{path}/X_train_encoded.npy")
        y_train = np.load(f"{path}/y_train.npy")
        X_val_encoded = np.load(f"{path}/X_val_encoded.npy")
        y_val = np.load(f"{path}/y_val.npy")
        X_test_encoded = np.load(f"{path}/X_test_encoded.npy")
        y_test = np.load(f"{path}/y_test.npy")
        with open(f"{path}/cat_info.pickle", "rb") as handle:
            cat_info = pkl.load(handle)
        with open(f"{path}/num_info.pickle", "rb") as handle:
            num_info = pkl.load(handle)
    return (
        X_train_encoded,
        y_train,
        X_val_encoded,
        y_val,
        X_test_encoded,
        y_test,
        cat_info,
        num_info,
    )
    # data = pd.read_csv(path)
    # print(data.shape)
    # print(data.head())


def build_mnist_ds(seed=42, info_path="models", batch_size=128):
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import random_split, DataLoader

    # Define your transformation: convert images to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Download the MNIST training and test datasets
    train_val_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Define the size of the validation set and compute training set size
    val_size = 10_000
    train_size = len(train_val_dataset) - val_size

    # Split the dataset into training and validation sets using a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_set, validation_set = random_split(
        train_val_dataset, [train_size, val_size], generator=generator
    )

    # Set the batch size
    batch_size = 64  # Adjust batch_size as needed

    # Create DataLoaders for each split
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class Loaders(NamedTuple):
    train: Any  # Replace Any with the appropriate type if available, e.g., DataLoader
    val: Any
    test: Any


def build_dataset(experiment="adult", batch_size=128, seed=42, info_path="models"):

    # check if the info path exists, else create a folder
    if not os.path.exists(info_path):
        os.makedirs(info_path)
    if not (experiment in ["mnist", "cifar_10"]):
        C = 2
        if experiment == "adult":
            X_train, y_train, X_val, y_val, X_test, y_test, enc_n, enc_d = (
                build_adult_ds(seed=seed, info_path=info_path)
            )
        elif experiment == "law":
            X_train, y_train, X_val, y_val, X_test, y_test, enc_n, enc_d = build_law_ds(
                seed=seed, info_path=info_path
            )
        elif experiment == "compas":
            X_train, y_train, X_val, y_val, X_test, y_test, enc_n, enc_d = (
                build_compas_ds(seed=seed, info_path=info_path)
            )
            C = 3
        elif experiment == "german_credit":
            X_train, y_train, X_val, y_val, X_test, y_test, enc_n, enc_d = (
                build_german_credit_ds(seed=seed, info_path=info_path)
            )
            C = 2
        elif experiment == "folktable_5":
            X_train, y_train, X_val, y_val, X_test, y_test, enc_n, enc_d = (
                build_folktable_5(seed=seed, info_path=info_path)
            )
            C = 5
        elif experiment == "ieee_cis_fraud":
            X_train, y_train, X_val, y_val, X_test, y_test, enc_n, enc_d = (
                build_iee_cis_fraud(seed=seed, info_path=info_path)
            )
        elif "ACS" in experiment:
            from folktables import (
                ACSIncome,
                ACSPublicCoverage,
                ACSMobility,
                ACSEmployment,
                ACSTravelTime,
            )

            if experiment == "ACSIncome":
                func = ACSIncome
            elif experiment == "ACSPublicCoverage":
                func = ACSPublicCoverage
            elif experiment == "ACSMobility":
                func = ACSMobility
            elif experiment == "ACSEmployment":
                func = ACSEmployment
            elif experiment == "ACSTravelTime":
                func = ACSTravelTime
            else:
                raise ValueError("Experiment not recognized")
            X_train, y_train, X_val, y_val, X_test, y_test, enc_n, enc_d = (
                build_folktable_task(func, save_path=f"data/{experiment}")
            )

        train_dataset = TensorDataset(
            torch.tensor(X_train).float(), torch.tensor(y_train)
        )
        val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val))
        test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        in_dim = X_train.shape[1]
    else:
        C = 10
        if experiment == "mnist":
            train_loader, val_loader, test_loader = build_mnist_ds(
                seed=seed, info_path=info_path, batch_size=batch_size
            )
            enc_n = None
            enc_d = None
            C = 10
            in_dim = 784
        elif experiment == "cifar_10":
            raise NotImplementedError("Cifar 10 not implemented")
        else:
            raise ValueError("Experiment not recognized")
    loaders = Loaders(train=train_loader, val=val_loader, test=test_loader)
    return loaders, enc_n, enc_d, C, in_dim


if __name__ == "__main__":
    print("hi")
    build_iee_cis_fraud()
