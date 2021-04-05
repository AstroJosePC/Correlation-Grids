from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype


def similar(a, b):
    # generalized to any a, b strings
    return SequenceMatcher(None, a, b).ratio()


def similarity_func(c, a):
    # specific function for error column matching
    if a.startswith('err'):
        return similar('err ' + c, a)
    else:
        return similar(c + ' err', a)


def _identify_errors(data: pd.DataFrame, col_set: set = None):
    columns = data.columns.values
    subset = col_set is not None
    error_columns = dict()
    for i, col in enumerate(columns):
        if subset and col not in col_set:
            continue
        low_col = col.lower()
        # Skip if it is error or string like data
        is_err = 'err' in low_col
        is_string = is_string_dtype(data[col])
        if is_err or is_string:
            continue

        # else first find adjacent columns
        idx = np.array([i - 2, i - 1, i + 1, i + 2])
        new_idx = idx[(idx >= 0) & (idx < columns.size)]
        adjacent = columns[new_idx]

        # find similarity for each adjacent column to col
        similarities = [similarity_func(low_col, adj.lower()) for adj in adjacent]
        err_idx = np.array(similarities).argmax()
        similarity_val = similarities[err_idx]
        err_col = adjacent[err_idx]
        if similarity_val > 0.60 and 'err' in err_col.lower():
            error_columns[col] = [err_col, similarity_val]

    return error_columns


def identify_errors(data: pd.DataFrame, col_set=None):
    column_matches = _identify_errors(data, col_set=col_set)

    # validate and drop duplicated error columns
    running_cols = set()
    max_col = dict()
    new_matches = dict()
    for col, (err_col, sim) in column_matches.items():
        if err_col not in running_cols:
            running_cols.add(err_col)
            max_col[err_col] = col, sim
            new_matches[col] = err_col
        else:
            if sim > max_col[err_col][1]:
                max_col[err_col] = col, sim
                new_matches[col] = err_col
    return new_matches
