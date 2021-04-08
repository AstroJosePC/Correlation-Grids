import copy
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from regplotter import regplot_log
from smart_grid import SmartGrid
from utils.grid_utils import identify_errors


def get_data(path: str):
    return pd.read_csv(path, sep=',', skipinitialspace=True, na_values=['#NAME?'])


def regplot_log_wrap(x, y, log_vars: Optional[list] = None, err_map: Optional[dict] = None,
                     data: Optional[pd.DataFrame] = None, ranges_map: Optional[dict] = None,
                     delta_map: Optional[dict] = None, seed: int = 123456, **kwargs):
    logx = x.name in log_vars
    logy = y.name in log_vars
    xerr = err_map.get(x.name)
    yerr = err_map.get(y.name)
    x_range = ranges_map.get(x.name)
    xdelta = delta_map.get(x.name)
    ydelta = delta_map.get(y.name)
    linmix_kws = dict(seed=seed)

    ax, plotter = regplot_log(data=data, x=x, y=y, xerr=xerr, yerr=yerr, logx=logx, logy=logy,
                              xdelta=xdelta, ydelta=ydelta, fit_xrange=x_range, linmix_kws=linmix_kws, **kwargs)


def nsq_grid(dataset: Union[pd.DataFrame, str], x_vars: list, y_vars: list, log_vars: Optional[list] = None,
             error_map: Optional[dict] = None, delta_map: Optional[dict] = None,
             regplot_kws: Optional[dict] = None, ann_coeff: bool = True, **kwargs):
    if isinstance(dataset, str):
        dataset = get_data(dataset)
    else:
        dataset = dataset.copy()
    all_vars = x_vars + y_vars
    error_map = parse_err_map(dataset, error_map, col_set=all_vars)
    ranges_map = {var: (np.nanmin(dataset[var]), np.nanmax(dataset[var])) for var in x_vars}

    regplot_kws = {} if regplot_kws is None else copy.copy(regplot_kws)
    regplot_kws.setdefault('ann_coeff', ann_coeff)

    # Default grid spacing aesthetics
    kwargs.setdefault('height', 2.0)
    kwargs.setdefault('aspect', 1.2)

    g = SmartGrid(dataset, x_vars=x_vars, y_vars=y_vars, log_vars=log_vars, **kwargs)
    g.map_offdiag(regplot_log_wrap, log_vars=log_vars, err_map=error_map, ranges_map=ranges_map,
                  delta_map=delta_map, data=dataset, linmix=True, **regplot_kws)

    # Call tight layout
    g.tight_layout()
    return g, error_map


def get_flux_upps(data: pd.DataFrame, error_map: Optional[dict] = None):
    cols = identify_flux(data)
    error_map = parse_err_map(data, error_map, col_set=cols)
    for col in cols:
        err_col = error_map[col]
        umask = data[err_col] > data[col]
        yield col, err_col, umask


def flux_prep(data: pd.DataFrame, error_map: Optional[dict] = None):
    """
    Pre-processing function for VISIR/Spitzer datasets

    :param data: dataset
    :param error_map: error map dictionary
    :return: delta map
    """
    dmap = {}
    for col, err_col, upp_mask in get_flux_upps(data, error_map):
        dmap[col] = (~upp_mask).astype(int)
        data.loc[upp_mask, col] = 2 * data[err_col][upp_mask]
    return dmap


def identify_flux(data: pd.DataFrame):
    return [col for col in data if _is_flux(col)]


def _is_flux(col: str):
    cl = col.lower()
    return 'err' not in cl and 'error' not in cl and cl.startswith('fl') or 'flux' in cl


def parse_err_map(data: pd.DataFrame, err_map: Optional[dict] = None, col_set: list = None):
    if err_map is None:
        return identify_errors(data, col_set)
    elif isinstance(err_map, dict):
        return err_map
    else:
        raise ValueError(f'Unexpected err_map input: {err_map} ')


def fluxes_grid(dataset: Union[pd.DataFrame, str], x_vars: list, y_vars: list, log_vars: Optional[list] = None,
                regplot_kws: Optional[dict] = None, ann_coeff: bool = True, **kwargs):
    if isinstance(dataset, str):
        dataset = get_data(dataset)
    else:
        dataset = dataset.copy()

    # Data Prep
    delta_map = flux_prep(dataset)
    # Generate Grid
    nsq_grid(dataset, x_vars, y_vars, log_vars=log_vars, delta_map=delta_map,
             regplot_kws=regplot_kws, ann_coeff=ann_coeff, **kwargs)
