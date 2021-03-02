import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from utils.stats import corr
from utils.misc import is_log
from CustomReg.regplotter import regplot_log
from UpperLimits.upper_prep import limit_arrays

sns.set_theme()
sns.set_style("ticks")
sns.set_context("paper")

err_dict = dict(FLC2H2='ERRC2H2', FLH2O_17='ERRH2O_17')


def regplot_log_filter(x, y, log_vars=None, data=None, markers=None, sizes=None, **kwargs):
    # x_vars, y_vars = get_vars(x, y, log_vars)
    # sns.regplot(x=x_vars, y=y_vars, **kwargs)
    # plt.gca().autoscale()
    logx = logy = False
    if x.name in log_vars:
        logx = True
    if y.name in log_vars:
        logy = True

    x_err_param = err_dict.get(x.name, None)
    y_err_param = err_dict.get(y.name, None)
    x_correct_params, y_correct_params, limits = limit_arrays(data, x.name, y.name, x_err_param, y_err_param)
    if limits is None:
        limits = np.full(x_correct_params.shape, 'norm')

    # scatter_kws = dict(size=limits, markers=markers, sizes=sizes)

    # regplot_log(x=x, y=y, logx=logx, logy=logy, scatter=False, **kwargs)

    sns.scatterplot(x=x_correct_params, y=y_correct_params, style=limits, hue=limits,
                    markers=markers, size=limits, sizes=sizes, legend=False, ax=plt.gca())


spitzer = pd.read_csv('Data/Spitzer_ALMA_sample.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])
variables = ['MSTAR', 'LSTAR', 'LOGLACC', 'LOGMACC', 'TEFF', 'FLH2O_17', 'N1331', 'FLC2H2', 'DIST', 'INCL']
plot_log = ['MSTAR', 'LSTAR', 'DIST', 'FLC2H2', 'FLH2O_17']

markers = {'west': r'$\leftarrow$', 'southwest': r'$\swarrow$', 'south': r'$\downarrow$', 'norm': 'o'}
sizes = {'west': 230, 'southwest': 230, 'south': 230, 'norm': 50}

# setup PairGrid, and plot data
g = sns.PairGrid(spitzer, dropna=True, vars=variables, diag_sharey=False, corner=True)
g.map_lower(regplot_log_filter, log_vars=plot_log, data=spitzer, markers=markers, sizes=sizes)

for ax in np.diag(g.axes):
    ax.set_visible(False)

# Map additional stuff
g.map_lower(corr, log_vars=plot_log)
g.map_lower(is_log, log_vars=plot_log)

# plt.show()

plt.savefig('output images/pair_plot_log+upp.png', dpi=150)
