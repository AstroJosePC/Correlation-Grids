import functools

import numpy as np
import pandas as pd


def add_stats(ax, color, corr_x, corr_y, plotter):
    # correlations
    rho_coeff = np.median(plotter._chain['corr'])
    rho_low, rho_high = np.quantile(plotter._chain['corr'], [0.1, 0.9]) - rho_coeff
    # slope
    beta = np.median(plotter._chain['beta'])
    beta_low, beta_high = np.quantile(plotter._chain['beta'], [0.1, 0.9]) - beta
    # slope
    sigma = np.sqrt(np.median(plotter._chain['sigsqr']))
    sigma_low, sigma_high = np.sqrt(np.quantile(plotter._chain['sigsqr'], [0.1, 0.9])) - sigma
    ann = fr'$\rho={rho_coeff:.2f}^{{+{rho_high:.2f}}}_{{{rho_low:.2f}}}$, '
    ann += fr'$\beta={beta:.2f}^{{+{beta_high:.2f}}}_{{{beta_low:.2f}}}$, '
    ann += fr'$\sigma={sigma:.2f}^{{+{+sigma_high:.2f}}}_{{{sigma_low:.2f}}}$'
    ax.text(s=ann, x=corr_x, y=corr_y, transform=ax.transAxes,
            bbox={'boxstyle': 'round', 'pad': 0.25, 'facecolor': color, 'edgecolor': 'gray', 'alpha': 0.4},
            ha='center', fontsize=8)


def add_detections(ax, color, plotter, x, y):
    num_det = plotter.ydelta.sum()
    total = plotter.ydelta.size
    ann = f'{num_det}/{total} dets'
    ax.text(s=ann, x=x, y=y, transform=ax.transAxes, ha='right', fontsize=8,
            bbox={'boxstyle': 'round', 'pad': 0.25, 'facecolor': color, 'edgecolor': 'gray', 'alpha': 0.6})


def add_labels(ax, kwargs, plotter):
    input_data = kwargs.get('data')
    x_data, y_data = plotter.scatter_data
    for ix, iy, name in zip(x_data, y_data, input_data['NAME'][x_data.index]):
        ax.text(s=name, x=ix, y=iy, fontsize=4)


def plotter_wrapper(func, **kws):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y = kwargs.pop('xys', (0.95, 0.90))
        corr_x, corr_y = kwargs.pop('corr_xys', (0.8, 0.05))
        color = kwargs.get('color', 'white')
        ax, plotter = func(*args, **kwargs)
        # x, y = plotter.scatter_data

        if kws.get('labels', False):
            # add disk labels
            add_labels(ax, kwargs, plotter)

        if plotter.ydelta is not None and any(plotter.ydelta == 0):
            # Add detections info text
            add_detections(ax, color, plotter, x, y)

        if kws.get('stats', False):
            # add regression stats
            add_stats(ax, color, corr_x, corr_y, plotter)

        # if kws.get('results', False):
        #     output_results(y.name)
        #
        return ax, plotter

    return wrapper


def remove_disks(data, remove_list):
    if remove_list:
        outliers_mask = np.zeros(len(data)).astype(bool)
        for disk in remove_list:
            outliers_mask |= data['NAME'] == disk
        data = data[~outliers_mask]
    return data


def wind_disk_design():
    mrks = ['^', 's']
    ho = ['winds', 'disk']
    ls = ['solid', 'dashed']
    psbs = {'xys': [(0.45, 1.1), (0.65, 1.1)],
            'corr_xys': [(0.5, 1.02), (0.5, 0.94)]}
    return mrks, ho, ls, psbs


def aggregate_components(dataset: pd.DataFrame):
    """
    this method was used to restructure the input dataset from CRIRES/iSHELL and combine comopnents.
    It is a one-time used code, but it is saved for archiving and re-using purposes. Which is why is at the bottom.
    :param dataset:
    :return:
    """
    single_comp = dataset['COMP'] == 'SC'
    broad_comp = dataset['COMP'] == 'BC'
    narrow_comp = dataset['COMP'] == 'NC'
    abs_comp = dataset['COMP'] == 'abs'

    nc_data = dataset[narrow_comp]
    bc_data = dataset[broad_comp]

    merged_comp = nc_data.merge(bc_data, on='NAME', suffixes=['_nc', '_bc'])

    flux_vars = ['V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX']
    delta_map = {}

    for fvar in flux_vars:
        is_upper_nc = merged_comp[f'{fvar}_ERR_nc'] > merged_comp[f'{fvar}_nc']
        is_upper_bc = merged_comp[f'{fvar}_ERR_bc'] > merged_comp[f'{fvar}_bc']

        is_upper_sum = merged_comp.loc[is_upper_nc & is_upper_bc, [f'{fvar}_ERR_nc', f'{fvar}_ERR_bc']].sum(axis=1)
        merged_comp.loc[is_upper_nc & is_upper_bc, [fvar, f'{fvar}_ERR']] = is_upper_sum

        xpartial_upper_sum = merged_comp.loc[(~is_upper_nc) & is_upper_bc, [f'{fvar}_nc', f'{fvar}_ERR_bc']].sum(axis=1)
        merged_comp.loc[(~is_upper_nc) & is_upper_bc, [fvar, f'{fvar}_ERR']] = xpartial_upper_sum

        ypartial_upper_sum = merged_comp.loc[is_upper_nc & (~is_upper_bc), [f'{fvar}_ERR_nc', f'{fvar}_bc']].sum(axis=1)
        merged_comp.loc[is_upper_nc & (~is_upper_bc), [fvar, f'{fvar}_ERR']] = ypartial_upper_sum

        not_upper = (~is_upper_nc) & (~is_upper_nc)
        not_upper_data = merged_comp[not_upper]
        merged_comp.loc[not_upper, fvar] = not_upper_data[[f'{fvar}_nc', f'{fvar}_bc']].sum(axis=1)
        merged_comp.loc[not_upper, f'{fvar}_ERR'] = np.sqrt(
            (not_upper_data[[f'{fvar}_ERR_nc', f'{fvar}_ERR_bc']] ** 2).sum(axis=1))

        is_nan = merged_comp[[f'{fvar}_nc', f'{fvar}_bc', f'{fvar}_ERR_nc', f'{fvar}_ERR_bc']].isna().any(axis='columns')
        merged_comp.loc[is_nan, [f'{fvar}', f'{fvar}_ERR']] = np.nan

        delta_map[fvar] = f'{fvar}_delta'
        # noinspection PyUnresolvedReferences
        merged_comp[f'{fvar}_delta'] = not_upper.astype(int)

    n = len(flux_vars) * 3
    columns = merged_comp.columns[:-n]
    # merged_comp[columns].T.duplicated()
    transpose = merged_comp[columns].T
    to_keep = ~transpose.duplicated(keep='first')
    total_keep = transpose.index[to_keep].append(merged_comp.columns[-n:])

    # DATASET REQUIRED FURTHER MANUAL PROCESSING AND CLEANING AFTER THIS OUTPUT
    return merged_comp[total_keep].replace(['NC', 'BC'], r'NC/BC')
