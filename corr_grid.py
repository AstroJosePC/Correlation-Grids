from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regplotter import regplot_log
from smart_grid import SmartGrid
from utils.grid_utils import identify_errors


def get_data(path: str):
    return pd.read_csv(path, sep=',', skipinitialspace=True, na_values=['#NAME?'])


def regplot_log_wrap(x, y, log_vars: Optional[list] = None, err_map: Optional[dict] = None,
                     data: Optional[pd.DataFrame] = None, ranges_map: Optional[dict] = None,
                     delta_map: Optional[dict] = None, **kwargs):
    logx = x.name in log_vars
    logy = y.name in log_vars
    xerr = err_map.get(x.name)
    yerr = err_map.get(y.name)
    x_range = ranges_map.get(x.name)
    xdelta = delta_map.get(x.name)
    ydelta = delta_map.get(y.name)
    linmix_kws = dict(seed=123456)

    regplot_log(data=data, x=x, y=y, xerr=xerr, yerr=yerr, logx=logx, logy=logy,
                xdelta=xdelta, ydelta=ydelta, fit_xrange=x_range, linmix_kws=linmix_kws, **kwargs)


def nsq_grid(dataset: Union[pd.DataFrame, str], x_vars: list, y_vars: list, log_vars: Optional[list] = None,
             grid_kws: Optional[dict] = None, regplot_kws: Optional[dict] = None):
    if isinstance(dataset, str):
        dataset = get_data(dataset)
    else:
        dataset = dataset.copy()
    all_vars = x_vars + y_vars
    error_map = identify_errors(dataset, col_set=all_vars)
    ranges_map = {var: (np.nanmin(dataset[var]), np.nanmax(dataset[var])) for var in all_vars}

    delta_map = {}
    for col, err_col in error_map.items():
        upp_mask = dataset[err_col] > dataset[col]
        delta_map[col] = (~upp_mask).astype(int)
        dataset.loc[upp_mask, col] = 2 * dataset[err_col][upp_mask]

    grid_kws = {} if grid_kws is None else copy.copy(grid_kws)
    regplot_kws = {} if regplot_kws is None else copy.copy(regplot_kws)
    g = SmartGrid(dataset, x_vars=x_vars, y_vars=y_vars, log_vars=log_vars, **grid_kws)
    g.map_offdiag(regplot_log_wrap, log_vars=log_vars, err_map=error_map, ranges_map=ranges_map,
                  delta_map=delta_map, data=dataset, linmix=True, **regplot_kws)
    return g, error_map


if __name__ == '__main__':
    # visir_gridkws = dict(hue='')

    # TEST WITH VISIR DATASET + UPPER LIMITS
    x_visir = ['Mstar', 'Teff']
    y_visir = ['flux_x', 'flux_y']
    visir_log = ['Mstar', 'Lstar', 'Teff', 'flux_x', 'flux_y', 'flux', 'fwhm_x', 'fwhm_y', 'fwhm']

    g1, err_map1 = nsq_grid('Data/VISIR_merged_fluxes_TMP.csv', x_visir, y_visir, log_vars=visir_log)
    plt.show()
    # TEST WITH VISIR DATASET + NO UPPER LIMITS
    y_visir = ['logLacc', 'Lstar']
    g2, err_map2 = nsq_grid('Data/VISIR_merged_fluxes_TMP.csv', x_visir, y_visir, log_vars=visir_log)
    plt.show()

    # TEST WITH SPITZER DATASET + UPPER LIMITS
    x_spitzer = ['MSTAR', 'TEFF']
    y_spitzer = ['FLH2O_17', 'FLHCN']
    spitzer_log = x_spitzer + y_spitzer
    g3, err_map3 = nsq_grid('Data/Spitzer_ALMA_sample.csv', x_spitzer, y_spitzer, log_vars=spitzer_log)
    plt.show()

    # TEST WITH SPITZER DATASET + NO UPPER LIMITS
    y_spitzer = ['LOGRDUST95', 'N1331']
    g4, err_map4 = nsq_grid('Data/Spitzer_ALMA_sample.csv', x_spitzer, y_spitzer, log_vars=spitzer_log)
    plt.show()
