from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype


def similar(a, b):
    # generalized to any a, b strings
    return SequenceMatcher(None, a, b).ratio()


def similar2err(c, a):
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
        # idx = np.array([i - 2, i - 1, i + 1, i + 2])
        idx = np.array([i + 1, i - 1, i + 2, i - 2])
        new_idx = idx[(idx >= 0) & (idx < columns.size)]
        adjacent = columns[new_idx]

        # find similarity for each adjacent column to col
        similarities = [similar2err(low_col, adj.lower()) for adj in adjacent]
        err_idx = np.array(similarities).argmax()
        similarity_val = similarities[err_idx]
        err_col = adjacent[err_idx]
        if similarity_val > 0.60 and 'err' in err_col.lower():
            error_columns[col] = [err_col, similarity_val]

    return error_columns


def _identify_errorsv2(data: pd.DataFrame, col_set: set = None):
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
        # idx = np.array([i - 2, i - 1, i + 1, i + 2])
        idx = np.array([i + 1, i - 1, i + 2, i - 2])
        new_idx = idx[(idx >= 0) & (idx < columns.size)]
        adjacent = columns[new_idx]

        # find similarity for each adjacent column to col
        similarities = np.array([similar2err(low_col, adj.lower()) for adj in adjacent])
        filtered_sims = similarities > 0.75
        is_err_cols = np.array(['err' in adj.lower() for adj in adjacent])
        potential_mask = is_err_cols & filtered_sims

        masked_sims = similarities[potential_mask]
        sims_argsort = masked_sims.argsort()[::-1]
        sorted_sims = masked_sims[sims_argsort]
        sorted_cols = np.array(adjacent)[potential_mask][sims_argsort]

        if len(sorted_sims) > 0:
            error_columns[col] = [*zip(sorted_cols, sorted_sims)]
    return error_columns


def identify_errors(data: pd.DataFrame, col_set: set = None):
    column_matches = _identify_errors(data, col_set=col_set)

    # validate and drop duplicated error columns
    max_col = dict()
    new_matches = dict()
    discarded_matches = dict()
    for col, (err_col, sim) in column_matches.items():
        if err_col not in max_col:
            # first time matching col - err_col
            max_col[err_col] = col, sim
            new_matches[col] = err_col
        # The next two statements cover duplicated error columns in our search
        # First we test whether the new col & error col match has higher similarity than the original match
        # IF so, we take the new match. The old match is then discarded
        elif sim > max_col[err_col][1]:
            old_col, old_sim = max_col[err_col]
            max_col[err_col] = col, sim
            new_matches[col] = err_col
            discarded_matches[old_col] = err_col, old_sim
        # Otherwise
    return new_matches


def identify_errorsv2(data: pd.DataFrame, col_set: set = None):
    matches = _identify_errorsv2(data, col_set=col_set)

    inverse_matches = dict()
    for col, values in matches.items():
        for err_col, sim in values:
            if err_col in inverse_matches:
                inverse_matches[err_col].append((col, sim))
            else:
                inverse_matches[err_col] = [(col, sim)]

    new_matches = dict()
    for err_col, inv_values in inverse_matches.items():
        inv_col, inv_sim = sorted(inv_values, key=lambda x: x[1], reverse=True)[0]
        if inv_col in new_matches:
            err_col2 = new_matches[inv_col]
            sim2 = [tup for tup in matches[inv_col] if tup[0] == err_col2][0][1]
            # inv_col2, inv_sim2 = sorted(inverse_matches[err_col], key=lambda x: x[1], reverse=True)[0]
            if inv_sim > sim2:
                new_matches[inv_col] = err_col
        else:
            new_matches[inv_col] = err_col

    return new_matches
