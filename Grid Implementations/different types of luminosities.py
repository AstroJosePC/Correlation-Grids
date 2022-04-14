import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from corr_grid import nsq_grid, regplot_log_wrap
from tools import plotter_wrapper, wind_disk_design, remove_disks

mpl.rcParams['figure.dpi'] = 200

# PARAMETERS
LABELS = True
STATS = True

# REMOVE FOLLOWING TARGETS:
low_mass = ['WaOph6', 'AA Tau']
outliers = ['MWC297', 'TW Hya']

# Other params
log_vars = ['MSTAR', 'LSTAR', 'TEFF', 'V1_LUM', 'W2_LUM', 'V1_NORM_LUM']
plotter_wrap = plotter_wrapper(regplot_log_wrap, labels=LABELS, stats=STATS)
labels_map = {'V1_LUM': r'CO Lum Proxy',
              'W2_LUM': r'W2 Luminosity',
              'V1_NORM_LUM': r'CO Continuum Normalized Lum',
              'FW10': r'FWHM 10%'}

if __name__ == '__main__':
    dataset = pd.read_csv('../Data/by shapes dataset + different lums.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])
    dataset = remove_disks(dataset, outliers + low_mass)

    xvars = ['MSTAR', 'LSTAR', 'LOGLACC', 'INCL', 'N1330', 'FW10']
    yvars = ['V1_LUM', 'W2_LUM', 'V1_NORM_LUM', 'FW10']

    scatter_kws = dict(alpha=0.7, zorder=2.5)
    line_kws = dict(alpha=0.5, linewidth=3)
    regplot_kws = dict(n_boot=1000, seed=31, size=70, ci=None, scatter_kws=scatter_kws, line_kws=line_kws)

    markers, hue_order, linestyles, pass_subsets = wind_disk_design()
    g, error_map = nsq_grid(dataset, hue='shape', hue_order=hue_order, palette='colorblind', x_vars=xvars, y_vars=yvars,
                            log_vars=log_vars, pass_subsets=pass_subsets, labels_map=labels_map, ann_coeff=False, copy=False,
                            qrange=(0, 1),
                            height=3, legend=True, regplot_kws=regplot_kws, markers=markers, linestyles=linestyles,
                            plotter=plotter_wrap)
    plt.gcf().savefig(r'new correlation grid-rename me.pdf')
