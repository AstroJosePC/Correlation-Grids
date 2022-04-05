import re
from copy import copy as copy_obj
from typing import Optional, List, Tuple
from typing import Union

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from regplotter import regplot_log
from smart_grid import SmartGrid
from utils.grid_utils import identify_errorsv2, similar

# Variable to assert these variables do not have errors; for development purposes
no_err_assert = ['Teff', 'Mstar', 'Lstar', 'logLacc', 'dist', 'incl', 'BC_fwhm', 'n_13-30',
                 'MSTAR', 'LSTAR', 'LOGLACC', 'LOGMACC', 'TEFF', 'N1331', 'FNIR', 'DIST', 'LOGRDUST95',
                 'RSUBL', 'RSNOW_ACC', 'LOGRDUST68']


def get_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=',', skipinitialspace=True, na_values=['#NAME?'])


def regplot_log_wrap(x, y, log_vars: Optional[list] = None, err_map: Optional[dict] = None,
                     data: Optional[pd.DataFrame] = None, delta_map: Optional[dict] = None, **kwargs):
    logx = x.name in log_vars
    logy = y.name in log_vars
    xerr = err_map.get(x.name)
    yerr = err_map.get(y.name)
    xdelta = delta_map.get(x.name)
    ydelta = delta_map.get(y.name)

    ax, plotter = regplot_log(data=data, x=x, y=y, xerr=xerr, yerr=yerr, logx=logx, logy=logy,
                              xdelta=xdelta, ydelta=ydelta, **kwargs)
    return ax, plotter


def nsq_grid(dataset: Union[pd.DataFrame, str], x_vars: list, y_vars: list, log_vars: Optional[list] = None,
             error_map: Optional[dict] = None, delta_map: Optional[dict] = None, regplot_kws: Optional[dict] = None,
             ann_coeff: bool = True, copy: bool = True, plotter: callable = None, legend: bool = False,
             qrange: Tuple = (0.025, 0.975), grid: Optional[SmartGrid] = None, **kwargs):
    if isinstance(dataset, str):
        dataset = get_data(dataset)
    elif copy:
        dataset = dataset.copy()
    if log_vars is None:
        log_vars = []

    all_vars = x_vars + y_vars
    y_vars = y_vars[::-1]
    assert len(set(all_vars) - set(dataset.columns)) == 0, 'A column included in x_vars or y_vars is not in the dataset'
    error_map = parse_err_map(dataset, error_map, col_set=all_vars)
    print(all_vars, error_map)

    if delta_map is None:
        delta_map = dict()

    regplot_kws = {} if regplot_kws is None else copy_obj(regplot_kws)
    regplot_kws.setdefault('ann_coeff', ann_coeff)
    regplot_kws.setdefault('legend', legend)
    regplot_kws.setdefault('qrange', qrange)

    if grid:
        assert len(grid.x_vars) == len(x_vars) and len(grid.y_vars) == len(y_vars), 'Input grid does not match the ' \
                                                                                    'intended mapping '
        g = grid
    else:
        # Default grid spacing aesthetics
        kwargs.setdefault('height', 2.0)
        kwargs.setdefault('aspect', 1.2)
        g = SmartGrid(dataset, x_vars=x_vars, y_vars=y_vars, log_vars=log_vars, **kwargs)
    if legend:
        g._extract_legend_handles = True
        g._legend_out = False

    if plotter is None:
        plotter = regplot_log_wrap
    g.map_offdiag(plotter, log_vars=log_vars, err_map=error_map, delta_map=delta_map, linmix=True, **regplot_kws)
    # Call tight layout
    if legend:
        g.add_legend(fontsize='xx-small', )
    g.tight_layout()
    return g, error_map


def _flux_upps(data: pd.DataFrame, error_map: Optional[dict] = None, subset: Optional[list] = None):
    cols = identify_flux(data, subset)
    error_map = parse_err_map(data, error_map, col_set=cols)
    for col in cols:
        err_col: str = error_map[col]
        umask: pd.Series = data[err_col] > data[col]
        yield col, err_col, umask


def _dist_col(data: pd.DataFrame):
    indx = np.argmax([similar('dist', b.lower()) for b in data.columns])
    return data.columns[indx]


def flux2lum(flux: pd.Series, dist: pd.Series, correction_col: Union[pd.Series, float] = 1.0) -> pd.Series:
    """
    Calculate luminosity in solar luminosities.
    The last term in the calculation is the unit conversion factor to Lsun.
    :param flux: flux in units of 10^-14 erg / cm^2 / s
    :param dist: distance in units of parsec
    :param correction_col:
    :return: luminosities in units of Lsun
    """
    return 4 * np.pi * flux * dist ** 2 * 2487.30567840084 * correction_col


def flux_prep(data: pd.DataFrame, x_vars: Optional[List[str]] = None, y_vars: Optional[List[str]] = None,
              log_vars: Optional[List[str]] = None, labels_map: Optional[dict] = None,
              error_map: Optional[dict] = None, correction_col: Optional[str] = None):
    """
    Pre-processing function for VISIR/Spitzer datasets

    :param correction_col:
    :param data: dataset
    :param x_vars:
    :param y_vars:
    :param log_vars:
    :param labels_map:
    :param error_map: error map dictionary
    :return: delta map
    """
    # Dictionary to store and map delta arrays
    dmap = {}
    # Get column that looks like distance
    dist = _dist_col(data)
    assert dist, 'Could not find distance column wtf?'
    # Get subset
    all_vars = []
    if x_vars is not None:
        all_vars += x_vars
    if y_vars is not None:
        all_vars += y_vars
    for col, err_col, upp_mask in _flux_upps(data, error_map, all_vars):
        # Generate new column name for luminosities
        lum_col = re.sub(r'[_ ]?fl(ux)?[_ ]?', _sub_lum, col, flags=re.I)
        lum_err_col = re.sub(r'[_ ]?fl(ux)?[_ ]?', _sub_lum, err_col, flags=re.I)

        # Store logical NOT of upper limit mask as delta array for linmix
        dmap[lum_col] = f'{lum_col}_delta'
        data[f'{lum_col}_delta'] = (~upp_mask).astype(int)
        # Assign upper limits from error column into data column using upp_mask
        data.loc[upp_mask, col] = data[err_col][upp_mask]

        # Calculate luminosity of line flux and set to data
        # Set luminosity error values to data
        # Since we usually have no errors for distance we can't propagate errors, so we assume distance is a scalar
        # TODO: Change method to at least search for distance errors!
        if correction_col in data:
            data[lum_col] = flux2lum(data[col], data[dist], correction_col=data[correction_col])
            data[lum_err_col] = flux2lum(data[err_col], data[dist], correction_col=data[correction_col])
        else:
            data[lum_col] = flux2lum(data[col], data[dist])
            data[lum_err_col] = flux2lum(data[err_col], data[dist])

        # Replace flux variables with luminosities to plot on grid
        if x_vars and col in x_vars:
            x_vars[x_vars.index(col)] = lum_col
        if y_vars and col in y_vars:
            y_vars[y_vars.index(col)] = lum_col
        if log_vars:
            log_vars.append(lum_col)
        if labels_map and col in labels_map:
            labels_map[lum_col] = labels_map.pop(col)
    return dmap


def _sub_lum(matchobj: re.Match):
    match_span = matchobj.span()
    string_span = matchobj.pos, matchobj.endpos
    if match_span == string_span:
        return 'Lum'
    elif match_span[0] == 0:
        return 'Lum_'
    elif match_span[1] == string_span[1]:
        return '_Lum'
    else:
        return '_Lum_'


def identify_flux(data: pd.DataFrame, subset: Optional[list] = None) -> List[str]:
    if subset:
        return [col for col in data if col in subset and not is_string_dtype(data[col]) and _is_flux(col)]
    else:
        return [col for col in data if _is_flux(col) if not is_string_dtype(data[col])]


def _is_flux(col: str):
    """
    Test weather col is a string that looks like a name given to a flux data column
    This test function is rather crude so I should expect bugs in the future
    :param col: string to test
    :return: True if test looks like a flux data column, False otherwise
    """
    cl = col.lower()
    return 'err' not in cl and 'error' not in cl and re.search(r'fl(ux)?', cl)


def parse_err_map(data: pd.DataFrame, err_map: Optional[dict] = None, col_set: list = None) -> dict:
    col_set = set(col_set) if col_set else None
    if err_map is None:
        return identify_errorsv2(data, col_set)
    elif isinstance(err_map, dict):
        new_set = set(col_set) - set(err_map.keys()) if col_set else None
        if col_set and len(new_set) > 0:
            new_map = identify_errorsv2(data, col_set=new_set)
            new_map.update(err_map)
            return new_map
        else:
            return err_map
    else:
        raise ValueError(f'Unexpected err_map input: {err_map} ')


def fluxes_grid(dataset: Union[pd.DataFrame, str], x_vars: list, y_vars: list, log_vars: Optional[list] = None,
                ann_coeff: bool = True, labels_map: dict = None, regplot_kws: Optional[dict] = None, **kwargs):
    """
    Correlation grid for VISIR/Spitzer table and flux processing format
    """
    if isinstance(dataset, str):
        dataset = get_data(dataset)
    else:
        dataset = dataset.copy()

    # Copy x_vars and y_var lists in case they can modified by flux_prep
    x_vars = copy_obj(x_vars)
    y_vars = copy_obj(y_vars)

    # Copy labels map in case they are modified in flux_prep
    labels_map = copy_obj(labels_map)

    # Flux Data Prep
    delta_map = flux_prep(dataset, x_vars, y_vars, log_vars, labels_map)
    # Generate Grid
    g, error_map = nsq_grid(dataset, x_vars, y_vars, log_vars=log_vars, delta_map=delta_map, regplot_kws=regplot_kws,
                            ann_coeff=ann_coeff, copy=False, labels_map=labels_map, **kwargs)
    return g
