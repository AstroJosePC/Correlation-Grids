import functools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corr_grid import nsq_grid, regplot_log_wrap, flux_prep

mpl.rcParams['figure.dpi'] = 200


# def plotter_detection_labels():
#     pass

def det_label_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y = kwargs.pop('xys', (0.95, 0.90))
        corr_x, corr_y = kwargs.pop('corr_xys', (0.8, 0.05))
        color = kwargs.get('color', 'white')
        ax, plotter = func(*args, **kwargs)
        # x, y = plotter.scatter_data
        if plotter.ydelta is None:
            return ax, plotter
        else:
            num_det = plotter.ydelta.sum()
            total = plotter.ydelta.size
            ann = f'{num_det}/{total} dets'
            ax.text(s=ann, x=x, y=y, transform=ax.transAxes, ha='right', fontsize=8,
                    bbox={'boxstyle': 'round', 'pad': 0.25, 'facecolor': color, 'edgecolor': 'gray', 'alpha': 0.6})

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
            return ax, plotter

    return wrapper


plotter_wrap = det_label_decorator(regplot_log_wrap)

if __name__ == '__main__':
    # dataset = pd.read_csv('../../Data/iSHELL_outputs_merged.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])
    dataset = pd.read_csv('../../Data/CRIRES&iSHELL_v1.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])

    single_comp = dataset['COMP'] == 'SC'
    broad_comp = dataset['COMP'] == 'BC'
    narrow_comp = dataset['COMP'] == 'NC'
    abs_comp = dataset['COMP'] == 'abs'

    v1ul_mask = dataset['V1_FLUX_ERR'] > dataset['V1_FLUX']
    v2ul_mask = dataset['V2_FLUX_ERR'] > dataset['V2_FLUX']

    v1det_subset = dataset[~v1ul_mask]
    v2det_subset = dataset[~v2ul_mask]

    v1sc_dets = dataset[single_comp & (~v1ul_mask)].copy()
    v1sc_dets['V1_FLUX_CORR'] = v1sc_dets['V1_FLUX'] * v1sc_dets['W2_JY']
    v1sc_dets['V1_FLUX_CORR_ERR'] = v1sc_dets['V1_FLUX_ERR'] * v1sc_dets['W2_JY']

    v1sc_dets['V1_CEN_CORR'] = v1sc_dets['V1_CEN'] - v1sc_dets['RV']

    # yvars = ['V1_FLUX_CORR', 'V1_FWHM', 'V1_CEN_CORR']
    # log_vars = ['V1_FLUX', 'V1_FLUX_CORR', 'MSTAR', 'LSTAR', 'TEFF']
    # labels_map = {'V1_FLUX_CORR': r'CO Lum (L$_\odot$)',
    #               'V1_FWHM': 'CO FWHM (km/s)',
    #               'V1_CEN_CORR': 'CO Cen (km/s)'}

    sc_all = dataset[single_comp].copy()
    dataset_copy = dataset[~abs_comp].copy()
    # dataset_copy.drop(index=4)
    xvars = ['MSTAR', 'LSTAR', 'LOGLACC', 'INCL', 'N1330']
    # xvars = ['MSTAR', 'LSTAR', 'LOGLACC']
    yvars = ['V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX']
    log_vars = ['MSTAR', 'LSTAR', 'TEFF', 'V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX']
    labels_map = {'V1_FLUX': r'CO v1-0 Lum (L$_\odot$)',
                  'V1H_FLUX': r'CO v1H-0H Lum (L$_\odot$)',
                  'V2_FLUX': r'CO v2-1 Lum (L$_\odot$)',
                  'I13_FLUX': r'$^{13}$CO v1-0 Lum (L$_\odot$)'}

    scatter_kws = dict(alpha=0.7, zorder=2.5)
    line_kws = dict(alpha=0.5, linewidth=3)
    regplot_kws = dict(n_boot=10000, seed=31, size=70, ci=None, scatter_kws=scatter_kws, line_kws=line_kws)
    markers = ['v', 's', 'o']
    linestyles = ['solid', 'dashed', 'dashdot']
    pass_subsets = {'xys': [(0.45, 1.1), (0.65, 1.1), (0.85, 1.1)],
                    'corr_xys': [(0.5, 1.02), (0.5, 0.94), (0.5, 0.86)]}
    delta_map = flux_prep(dataset_copy, y_vars=yvars, log_vars=log_vars, labels_map=labels_map, correction_col='W2_JY')
    g, error_map = nsq_grid(dataset_copy, hue='COMP', palette='colorblind', x_vars=xvars, y_vars=yvars,
                            log_vars=log_vars, pass_subsets=pass_subsets,
                            delta_map=delta_map, labels_map=labels_map, ann_coeff=False, copy=False, qrange=(0, 1),
                            height=3, legend=True, regplot_kws=regplot_kws, markers=markers, linestyles=linestyles,
                            plotter=plotter_wrap)
    # plt.gcf().savefig('COV1_fluxes_v2.pdf')
    # delta_map = flux_prep(v1sc_dets, xvars, yvars, log_vars, labels_map=labels_map)
    # delta_map = None
    # err_map = {'TEFF': 'TEFF_ERR', 'V1_Lum_CORR': 'V1_Lum_CORR_ERR'}
    # g, error_map = nsq_grid(v1sc_dets, xvars, yvars, log_vars=log_vars, delta_map=delta_map,
    #                         labels_map=labels_map, ann_coeff=True, copy=False, qrange=(0, 1))

    # # plt.gcf().savefig('COV1_fluxes.pdf')

    # log_vars = ['V1_FLUX', 'V1_FLUX_CORR', 'MSTAR', 'LSTAR', 'TEFF']
    # # delta_map = flux_prep(v1sc_dets, xvars, yvars, log_vars, labels_map=labels_map)
    # g, error_map = nsq_grid(v1sc_dets, xvars, yvars, log_vars=log_vars, delta_map=delta_map,
    #                         labels_map=labels_map, ann_coeff=True, copy=False, qrange=(0, 1))
    # plt.gcf().savefig('COV1_fluxes_v2.pdf')
    plt.show()
