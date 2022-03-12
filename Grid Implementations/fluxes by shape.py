import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from corr_grid import nsq_grid, regplot_log_wrap, _flux_upps
from tools import plotter_wrapper

mpl.rcParams['figure.dpi'] = 200

LABELS = True
STATS = True

# REMOVE FOLLOWING TARGETS:
low_mass = ['WaOph6', 'AA Tau']
outliers = ['MWC297']

# Other params
log_vars = ['MSTAR', 'LSTAR', 'TEFF', 'W2_JY', 'V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX']
labels_map = {'V1_FLUX': r'CO v1-0 Flux',
              'V1H_FLUX': r'CO v1H-0H Flux',
              'V2_FLUX': r'CO v2-1 Flux',
              'I13_FLUX': r'$^{13}$CO v1-0 Flux'}

plotter_wrap = plotter_wrapper(regplot_log_wrap, labels=LABELS, stats=STATS)

if __name__ == '__main__':
    dataset = pd.read_csv('../Data/shapes dataset v2.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])
    # dataset = remove_disks(dataset, outliers)

    xvars = ['MSTAR', 'LSTAR', 'LOGLACC', 'INCL', 'N1330']
    yvars = ['W2_JY', 'V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX']

    scatter_kws = dict(alpha=0.7, zorder=2.5)
    line_kws = dict(alpha=0.5, linewidth=3)
    regplot_kws = dict(n_boot=10000, seed=31, size=70, ci=None, scatter_kws=scatter_kws, line_kws=line_kws)

    delta_map = {}
    for col, err_col, upp_mask in _flux_upps(dataset, None, yvars):
        # Store logical NOT of upper limit mask as delta array for linmix
        delta_map[col] = f'{col}_delta'
        dataset[f'{col}_delta'] = (~upp_mask).astype(int)
        # Assign upper limits from error column into dataset column using upp_mask
        dataset.loc[upp_mask, col] = dataset[err_col][upp_mask]

    markers = ['^', 's']
    hue_order = ['winds', 'disk']
    linestyles = ['solid', 'dashed']
    pass_subsets = {'xys': [(0.45, 1.1), (0.65, 1.1)],
                    'corr_xys': [(0.5, 1.02), (0.5, 0.94)]}
    g, error_map = nsq_grid(dataset, hue='shape', hue_order=hue_order, palette='colorblind', x_vars=xvars,
                            y_vars=yvars, log_vars=log_vars, pass_subsets=pass_subsets,
                            delta_map=delta_map, labels_map=labels_map, ann_coeff=False, copy=False, qrange=(0, 1),
                            height=3, legend=True, regplot_kws=regplot_kws, markers=markers, linestyles=linestyles,
                            plotter=plotter_wrap)
    plt.gcf().savefig(r'new correlation grid-rename me.pdf')
