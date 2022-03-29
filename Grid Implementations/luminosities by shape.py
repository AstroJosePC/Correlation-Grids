import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from corr_grid import nsq_grid, regplot_log_wrap, flux_prep
from tools import plotter_wrapper, wind_disk_design, remove_disks

mpl.rcParams['figure.dpi'] = 200

# PARAMETERS
LABELS = True
STATS = True

# REMOVE FOLLOWING TARGETS:
low_mass = ['WaOph6', 'AA Tau']
outliers = ['MWC297', 'TW Hya']

# Additional columns
add_xvars = ['TEFF', 'FW10']
add_yvars = ['FW10']

# Other params
log_vars = ['MSTAR', 'LSTAR', 'TEFF', 'V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX']
plotter_wrap = plotter_wrapper(regplot_log_wrap, labels=LABELS, stats=STATS)
labels_map = {'V1_FLUX': r'CO v1-0 Lum (L$_\odot$)',
              'V1H_FLUX': r'CO v1H-0H Lum (L$_\odot$)',
              'V2_FLUX': r'CO v2-1 Lum (L$_\odot$)',
              'I13_FLUX': r'$^{13}$CO v1-0 Lum (L$_\odot$)'}

if __name__ == '__main__':
    dataset = pd.read_csv('../Data/shapes dataset v2.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])
    dataset = remove_disks(dataset, outliers + low_mass)

    xvars = ['MSTAR', 'LSTAR', 'LOGLACC', 'INCL', 'N1330'] + add_xvars
    yvars = ['V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX'] + add_yvars

    scatter_kws = dict(alpha=0.7, zorder=2.5)
    line_kws = dict(alpha=0.5, linewidth=3)
    regplot_kws = dict(n_boot=10000, seed=31, size=70, ci=None, scatter_kws=scatter_kws, line_kws=line_kws)

    markers, hue_order, linestyles, pass_subsets = wind_disk_design()
    delta_map = flux_prep(dataset, y_vars=yvars, log_vars=log_vars, labels_map=labels_map, correction_col='W2_JY')
    g, error_map = nsq_grid(dataset, hue='shape', hue_order=hue_order, palette='colorblind', x_vars=xvars, y_vars=yvars,
                            log_vars=log_vars, pass_subsets=pass_subsets,
                            delta_map=delta_map, labels_map=labels_map, ann_coeff=False, copy=False, qrange=(0, 1),
                            height=3, legend=True, regplot_kws=regplot_kws, markers=markers, linestyles=linestyles,
                            plotter=plotter_wrap)
    plt.gcf().savefig(r'new correlation grid-rename me.pdf')
