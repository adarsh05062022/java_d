import json
import os
import pickle
from typing import Dict, Union

import numpy as np
import sklearn.preprocessing
from pandas import DataFrame
from relbench.base import Database
from scipy.spatial.distance import cdist


def read_json(path: str):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data


def read_pickle(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data, path: str):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def sort_columns(
    df: DataFrame, domain: Dict[str, Dict[str, Union[str, int]]]
) -> DataFrame:
    """Sort columns of a DataFrame such that ID columns are first, then numerical features, then categorical features."""
    numerical_columns, categorical_columns = [], []
    for col, info in domain.items():
        if info["type"] == "continuous":
            numerical_columns.append(col)
        elif info["type"] == "discrete":
            categorical_columns.append(col)
        else:
            raise ValueError(f"Unknown feature type {info['type']}.")

    # Keep the frame's own order for the id columns. The upstream form built this list with
    # `list(set(df) - ...)`, and set iteration order over strings varies between processes, so
    # the id columns landed in a different order on every run -- two runs of the same seed wrote
    # CSVs whose headers differed (`loan_id,account_id` vs `account_id,loan_id`). Values were
    # unaffected, but the output was not reproducible. Consumers must still match on column
    # name, never position.
    id_columns = [
        col
        for col in df.columns
        if col not in set(numerical_columns) | set(categorical_columns)
    ]
    sorted_columns = (
        id_columns  # id columns, in the frame's own order
        + numerical_columns  # then numerical columns
        + categorical_columns  # then categorical columns
    )
    return df[sorted_columns]


def get_col_to_stype_dict(
    database: Database,
    data_path: str,
) -> Dict[str, Dict[str, str]]:
    type_to_stype = {
        "discrete": "categorical",
        "continuous": "numerical",
    }
    col_to_stype_dict = {}
    for table_name, table in database.table_dict.items():
        domain_path = os.path.join(data_path, f"{table_name}_domain.json")
        domain = read_json(domain_path)
        col_to_stype_dict[table_name] = {}
        for col in table.df.columns:
            if col.endswith("_id"):
                # TODO: should it be categorical or numerical?
                col_to_stype_dict[table_name][col] = "categorical"
            else:
                col_to_stype_dict[table_name][col] = type_to_stype[domain[col]["type"]]

    return col_to_stype_dict


# TODO: generalize to more normalization functions
# for now, we only need QuantileTransformer, so it's fine
def numerical_columns(data_path: str, table_name: str, df_columns) -> list:
    """Continuous (non-id) columns of a table, in the frame's own order."""
    domain = read_json(os.path.join(data_path, f"{table_name}_domain.json"))
    return [
        col
        for col in df_columns
        if not col.endswith("_id") and domain.get(col, {}).get("type") == "continuous"
    ]


def preprocess_database(
    database: Database,
    normalizers_cache_dir: str,
    seed: int = 0,
    data_path: str = None,
    categorical_encoding: str = "ordinal",
    onehot_layout: bool = False,
):
    """Preprocess a database by normalizing its data in-place.

    Under the default 'ordinal' encoding every non-id column is quantile-transformed, categorical
    codes included. That is part of why rare classes vanish (fix S4): `QuantileTransformer` maps a
    class to a band of the normal proportional to its frequency, so a 4.7%-mass class gets a
    correspondingly narrow target for the diffusion to land in.

    When `onehot_layout` is set -- either `categorical_encoding='onehot'` (fix S4) or
    `--categorical_diffusion multinomial` -- the categorical columns are left alone here and
    expanded to indicator columns when the graph is built, so every class gets an equal-width
    target and the diffusion process handles them directly (Gaussian over the simplex for S4,
    true multinomial for the discrete-diffusion path).
    """
    onehot_layout = onehot_layout or categorical_encoding == "onehot"
    for table_name, table in database.table_dict.items():
        df = table.df
        cols_no_id = [col for col in df.columns if not col.endswith("_id")]
        if onehot_layout:
            if data_path is None:
                raise ValueError("data_path is required for the one-hot / multinomial layout")
            cols_no_id = numerical_columns(data_path, table_name, df.columns)
            if not cols_no_id:
                continue
        data_raw = df[cols_no_id].to_numpy()
        data_processed = normalize(
            X=data_raw,
            cache_dir=normalizers_cache_dir,
            table_name=table_name,
            seed=seed,
        )
        df[cols_no_id] = data_processed


def normalize(
    X: np.ndarray,
    cache_dir: str,
    table_name: str,
    seed: int = 0,
):
    os.makedirs(cache_dir, exist_ok=True)
    normalizer_path = os.path.join(cache_dir, f"{table_name}_normalizer.pkl")

    if os.path.exists(normalizer_path):
        # normalizer has already been fitted
        print(f"Using cached normalizer for table {table_name}.")
        normalizer = read_pickle(normalizer_path)
    else:
        print(f"Fitting normalizer from scratch for table {table_name}.")
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
        normalizer.fit(X)
        write_pickle(normalizer, normalizer_path)

    X_transformed = normalizer.transform(X)
    return X_transformed


def denormalize_data(
    node_type_to_X_syn: Dict[str, np.ndarray],
    normalizers_dir: str,
    node_type_to_num_numerical_features: Dict[str, int] = None,
):
    """Invert the quantile transform.

    Under one-hot encoding (fix S4) the normalizer was fitted on the numerical columns alone, so
    only that block is inverted and the indicator columns pass through untouched.
    """
    node_type_to_X_denorm = {}
    for node_type, X_syn in node_type_to_X_syn.items():
        if node_type_to_num_numerical_features is None:
            node_type_to_X_denorm[node_type] = denormalize(
                X=X_syn, cache_dir=normalizers_dir, table_name=node_type
            )
            continue

        n_num = node_type_to_num_numerical_features[node_type]
        if n_num == 0:
            node_type_to_X_denorm[node_type] = X_syn
            continue
        num_block = denormalize(
            X=X_syn[:, :n_num], cache_dir=normalizers_dir, table_name=node_type
        )
        node_type_to_X_denorm[node_type] = np.concatenate(
            [num_block, X_syn[:, n_num:]], axis=1
        )
    return node_type_to_X_denorm


def denormalize(
    X: np.ndarray,
    cache_dir: str,
    table_name: str,
):
    normalizer_path = os.path.join(cache_dir, f"{table_name}_normalizer.pkl")
    if not os.path.exists(normalizer_path):
        raise ValueError(f"Normalizer for table {table_name} not found.")
    print(f"Using cached normalizer for table {table_name}.")
    normalizer = read_pickle(normalizer_path)
    X_denormalized = normalizer.inverse_transform(X)
    return X_denormalized


def extract_attributes(database: Database) -> Dict[str, np.ndarray]:
    table_name_to_X = {}
    for table_name, table in database.table_dict.items():
        df = table.df
        cols_no_id = [col for col in df.columns if not col.endswith("_id")]
        table_name_to_X[table_name] = df[cols_no_id].to_numpy()
    return table_name_to_X


def decode_categorical_matched(x_syn_col: np.ndarray, x_real_col: np.ndarray) -> np.ndarray:
    """Decode one categorical column so its marginal matches the real one (fix S1).

    Categorical columns are diffused as ordinal continuous values, so decoding with
    `np.round` thresholds at the midpoints between codes. That systematically erases a rare
    class whose code sits between two dense ones -- `loan.status` B (code 1, 4.7% of rows)
    lies between A (0) and C (2) and collapses to 1-6 rows, while D (code 3) sits at the edge
    of the range and survives.

    Instead of assuming the midpoints are the right thresholds, keep the ordering the model
    produced and place the thresholds where the real class proportions say they belong. Codes
    absent from the real data receive no mass, so gaps in the label encoding are preserved.
    """
    real_vals, real_counts = np.unique(x_real_col, return_counts=True)
    n = len(x_syn_col)

    # Largest-remainder apportionment so the assigned counts sum to exactly n.
    target = real_counts / real_counts.sum() * n
    counts = np.floor(target).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        counts[np.argsort(-(target - counts))[:remainder]] += 1

    order = np.argsort(x_syn_col, kind="stable")
    out = np.empty(n, dtype=int)
    pos = 0
    for val, count in zip(real_vals, counts):
        out[order[pos : pos + count]] = int(val)
        pos += count
    return out


def decode_onehot_synthetic_data(
    node_type_to_X_syn: Dict[str, np.ndarray],
    node_type_to_num_numerical_features: Dict[str, int],
    node_type_to_cat_num_classes: Dict[str, list],
) -> Dict[str, np.ndarray]:
    """Collapse one-hot blocks back to class codes by argmax (fix S4).

    Inverse of the expansion in `make_pkey_fkey_graph`. The numerical block passes through
    untouched; each categorical column is recovered as the argmax of its own block, so no class
    can be reached by drifting across a neighbouring one.
    """
    out = {}
    for table_name, X_syn in node_type_to_X_syn.items():
        n_num = node_type_to_num_numerical_features[table_name]
        num_block = np.round(X_syn[:, :n_num], 6)
        codes, offset = [], n_num
        for n_classes in node_type_to_cat_num_classes[table_name]:
            codes.append(X_syn[:, offset : offset + n_classes].argmax(axis=1))
            offset += n_classes
        if codes:
            out[table_name] = np.concatenate(
                [num_block, np.stack(codes, axis=1).astype(num_block.dtype)], axis=1
            )
        else:
            out[table_name] = num_block
    return out


def round_synthetic_data(
    node_type_to_X_syn: Dict[str, np.ndarray],
    node_type_to_X_real: Dict[str, np.ndarray],
    node_type_to_num_numerical_features: Dict[str, int],
    categorical_decoding: str = "round",
) -> Dict[str, np.ndarray]:
    for table_name in node_type_to_X_syn:
        X_syn = node_type_to_X_syn[table_name]
        X_real = node_type_to_X_real[table_name]
        num_numerical_features = node_type_to_num_numerical_features[table_name]

        X_syn_num = X_syn[:, :num_numerical_features]
        X_syn_cat = X_syn[:, num_numerical_features:]

        X_real_num = X_real[:, :num_numerical_features]
        X_real_cat = X_real[:, num_numerical_features:]

        # decode categorical features
        if categorical_decoding == "match":
            X_syn_cat = np.stack(
                [
                    decode_categorical_matched(X_syn_cat[:, col], X_real_cat[:, col])
                    for col in range(X_syn_cat.shape[1])
                ],
                axis=1,
            ) if X_syn_cat.shape[1] else X_syn_cat.astype(int)
        else:
            X_syn_cat = np.round(X_syn_cat).astype(int)
        # TODO: clip between smallest and largest class label
        # TODO: using sklear.LabelEncoder might be helpful for this purpose as it stores the number of classes
        # for now just check and manually correct
        for col in range(X_syn_cat.shape[1]):
            real_unique = np.unique(X_real_cat[:, col])
            # Clip to the largest class label rather than assuming the label encoding is dense.
            # The original `assert real_unique == arange(len(real_unique))` holds only when every
            # code 0..K-1 occurs in the training rows, which is true of the shipped datasets but
            # not of a subsample: the berka_split holdout removes every client with birth-year
            # code 73, leaving a gap. When the codes are contiguous this is identical to the
            # previous behaviour, since then max == len - 1.
            assert real_unique.min() >= 0
            X_syn_cat[:, col] = np.clip(X_syn_cat[:, col], 0, int(real_unique.max()))

            syn_unique = np.unique(X_syn_cat[:, col])
            if (
                len(syn_unique) != len(real_unique)
                or not (syn_unique == real_unique).all()
            ):
                print(
                    f"A categorical column's unique values do not match between synthetic and real data for table {table_name}, column {col}"
                )

        # round numerical features
        # X_syn_num = X_syn_num.astype(int)
        X_syn_num = np.round(X_syn_num, 6)

        disc_cols = []
        for col in range(X_real_num.shape[1]):
            uniq_vals = np.unique(X_real_num[:, col])
            # in clava it is: if len(uniq_vals) <= 32
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print(
            "Discrete cols (<= 32 integer categories) modeled as numerical cols:",
            disc_cols,
        )
        if len(disc_cols):
            X_syn_num = round_columns(X_real_num, X_syn_num, disc_cols)

        for col in range(X_syn_num.shape[1]):
            syn_unique = np.unique(X_syn_num[:, col])
            real_unique = np.unique(X_real_num[:, col])
            if (
                len(syn_unique) != len(real_unique)
                or not (syn_unique == real_unique).all()
            ):
                print(
                    f"A numerical column's unique values do not match between synthetic and real data for table {table_name}, column {col}"
                )

        node_type_to_X_syn[table_name] = np.concatenate((X_syn_num, X_syn_cat), axis=1)

    return node_type_to_X_syn


def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:, col])
        dist = cdist(
            X_synth[:, col][:, np.newaxis].astype(float),
            uniq[:, np.newaxis].astype(float),
        )
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth


def index_columns(df: DataFrame) -> DataFrame:
    """Index columns of a DataFrame such that each column is prefixed with its index."""
    cols_no_id = [col for col in df.columns if not col.endswith("_id")]
    num_digits = len(str(len(cols_no_id) - 1))
    col_name_mapper = {
        col: f"{i:0{num_digits}d}_{col}" for i, col in enumerate(cols_no_id)
    }
    return df.rename(columns=col_name_mapper)


def get_feature_types_count(db: Database, data_path: str):
    feature_type_to_count_dict = {}
    for table_name in db.table_dict:
        feature_type_to_count_dict[table_name] = {
            "numerical": 0,
            "categorical": 0,
        }
        domain_path = os.path.join(data_path, f"{table_name}_domain.json")
        domain = read_json(domain_path)
        for col in domain:
            if domain[col]["type"] == "continuous":
                feature_type_to_count_dict[table_name]["numerical"] += 1
            elif domain[col]["type"] == "discrete":
                feature_type_to_count_dict[table_name]["categorical"] += 1
            else:
                raise ValueError(f"Unknown feature type {domain[col]['type']}.")
    return feature_type_to_count_dict


def write_database(
    database: Database,
    save_dir: str,
    suffix: str = "",
):
    os.makedirs(save_dir, exist_ok=True)
    for table_name, table in database.table_dict.items():
        table.df.to_csv(
            os.path.join(save_dir, f"{table_name}{suffix}.csv"), index=False
        )
