import functools

import matplotlib as mpl
import numpy as np
import pandas as pd

from corr_grid import nsq_grid, regplot_log_wrap, _flux_upps

mpl.rcParams['figure.dpi'] = 200


# def plotter_detection_labels():
#     pass

def plotter_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y = kwargs.pop('xys', (0.95, 0.90))
        corr_x, corr_y = kwargs.pop('corr_xys', (0.8, 0.05))
        color = kwargs.get('color', 'white')
        ax, plotter = func(*args, **kwargs)
        # x, y = plotter.scatter_data

        data = kwargs.get('data')
        x_data, y_data = plotter.scatter_data
        for ix, iy, name in zip(x_data, y_data, data['NAME'][x_data.index]):
            ax.text(s=name, x=ix, y=iy, fontsize=4)

        if plotter.ydelta is not None:
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


plotter_wrap = plotter_wrapper(regplot_log_wrap)

if __name__ == '__main__':
    dataset = pd.read_csv('../Data/CRIRES&iSHELL_v1.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])

    single_comp = dataset['COMP'] == 'SC'
    broad_comp = dataset['COMP'] == 'BC'
    narrow_comp = dataset['COMP'] == 'NC'
    abs_comp = dataset['COMP'] == 'abs'

    dataset_noabs = dataset[~abs_comp].copy()

    # dataset to use...
    data = dataset_noabs

    xvars = ['MSTAR', 'LSTAR', 'LOGLACC', 'INCL', 'N1330']
    yvars = ['W2_JY', 'V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX']
    log_vars = ['MSTAR', 'LSTAR', 'TEFF', 'W2_JY', 'V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX']
    labels_map = {'V1_FLUX': r'CO v1-0 Flux',
                  'V1H_FLUX': r'CO v1H-0H Flux',
                  'V2_FLUX': r'CO v2-1 Flux',
                  'I13_FLUX': r'$^{13}$CO v1-0 Flux'}

    scatter_kws = dict(alpha=0.7, zorder=2.5)
    line_kws = dict(alpha=0.5, linewidth=3)
    regplot_kws = dict(n_boot=10000, seed=31, size=70, ci=None, scatter_kws=scatter_kws, line_kws=line_kws)

    hue_order = ['SC', 'NC', 'BC']
    markers = ['v', 's', 'o']
    linestyles = ['solid', 'dashed', 'dashdot']
    pass_subsets = {'xys': [(0.40, 1.18), (0.62, 1.18), (0.85, 1.18)],
                    'corr_xys': [(0.5, 1.08), (0.5, 0.98), (0.5, 0.88)]}

    delta_map = {}
    for col, err_col, upp_mask in _flux_upps(data, None, yvars):
        # Store logical NOT of upper limit mask as delta array for linmix
        delta_map[col] = f'{col}_delta'
        data[f'{col}_delta'] = (~upp_mask).astype(int)
        # Assign upper limits from error column into data column using upp_mask
        data.loc[upp_mask, col] = data[err_col][upp_mask]

    # g, error_map = nsq_grid(data, hue='COMP', hue_order=hue_order, palette='colorblind', x_vars=xvars,
    #                         y_vars=yvars, log_vars=log_vars, pass_subsets=pass_subsets,
    #                         delta_map=delta_map, labels_map=labels_map, ann_coeff=False, copy=False, qrange=(0, 1),
    #                         height=3, legend=True, regplot_kws=regplot_kws, markers=markers, linestyles=linestyles,
    #                         plotter=plotter_wrap)
    # plt.gcf().savefig(r'G:\My Drive\AcademicLife\Banzatti\Research\Seaborn Grids\seaborn_grids\output images\raw '
    #                   r'fluxes and W2\no absorption by component + labels.pdf')
    # print('----------------------- finalized first one -------------------------')
    markers = ['^', 's']
    hue_order = ['winds', 'disk']
    linestyles = ['solid', 'dashed']
    pass_subsets = {'xys': [(0.45, 1.1), (0.65, 1.1)],
                    'corr_xys': [(0.5, 1.02), (0.5, 0.94)]}
    g, error_map = nsq_grid(data, hue='shape', hue_order=hue_order, palette='colorblind', x_vars=xvars,
                            y_vars=yvars, log_vars=log_vars, pass_subsets=pass_subsets,
                            delta_map=delta_map, labels_map=labels_map, ann_coeff=False, copy=False, qrange=(0, 1),
                            height=3, legend=True, regplot_kws=regplot_kws, markers=markers, linestyles=linestyles,
                            plotter=plotter_wrap)
    # plt.gcf().savefig(r'G:\My Drive\AcademicLife\Banzatti\Research\Seaborn Grids\seaborn_grids\output images\raw '
    #                   r'fluxes and W2\no absorption by shape + labels.pdf')
    print('----------------------- finalized second one -------------------------')
