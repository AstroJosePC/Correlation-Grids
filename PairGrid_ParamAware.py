"""
Corner plots for parameter spaces. This time our correlation calculations can be applied to log data or nah

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

sns.set_theme()
sns.set_context("paper")


def get_vars(x, y, log_vars=None):
    if x.name in log_vars:
        x_vars = np.log10(x)
    else:
        x_vars = x
    if y.name in log_vars:
        y_vars = np.log10(y)
    else:
        y_vars = y
    return x_vars, y_vars


# Function to calculate correlation coefficient between two arrays
def corr(x, y, log_vars=None, **kwargs):
    # Modified from
    # https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
    # Calculate the value
    x_vars, y_vars = get_vars(x, y, log_vars)
    coeff = np.corrcoef(x_vars, y_vars)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coeff, 2))

    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)


def is_log(x, y, log_vars=None, **kwargs):
    # Map log scales

    # Get current axis
    ax = plt.gca()

    if x.name in log_vars:
        ax.set_xscale('log')
    if y.name in log_vars:
        ax.set_yscale('log')


def regplot_log(x, y, log_vars=None, **kwargs):
    # x_vars, y_vars = get_vars(x, y, log_vars)
    x_vars, y_vars = x, y
    if x.name in log_vars:
        logx = True
    else:
        logx = False
    sns.regplot(x=x_vars, y=y_vars, logx=logx, **kwargs)


spitzer = pd.read_csv('Data/Spitzer_ALMA_sample.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])

spitzer['Rco / Rsubl'] = spitzer['CORAD'] / spitzer['RSUBL']
spitzer['Rco / Rsnow'] = spitzer['CORAD'] / spitzer['RSNOW_ACC']
spitzer['R68%'] = 10 ** spitzer['LOGRDUST68']
spitzer['Rdust'] = 10 ** spitzer['LOGRDUST95']

# variables = ['MSTAR', 'LSTAR', 'LOGLACC', 'LOGMACC', 'TEFF', 'N1331', 'FNIR', 'DIST', 'INCL', 'F09MM',
#              'F13MM', 'R68%', 'Rdust', 'COV2V1', 'CORAD', 'RSUBL', 'Rco / Rsubl', 'Rco / Rsnow',
#              'FLCO_BC', 'FLCO_NC', 'FLH2O_17', 'FLHCN', 'FLC2H2', 'FLCO2', 'CONT_17', 'FLH2_17']

variables = ['MSTAR', 'LSTAR', 'LOGLACC', 'LOGMACC', 'TEFF', 'N1331', 'DIST', 'INCL']
plot_log = ['MSTAR', 'LSTAR', 'DIST']
# plot_log = [True, True, False, False, False, False, True, False]

# setup PairGrid, and plot data
g = sns.PairGrid(spitzer, dropna=True, vars=variables, diag_sharey=False, corner=True)
g.map_lower(sns.regplot, n_boot=4000)
g.map_diag(sns.histplot)

# Map additional stuff
g.map_lower(corr, log_vars=plot_log)
# g.map_lower(is_log, log_vars=plot_log)

# plt.show()

# indices = zip(*np.tril_indices_from(g.axes, -1))
# g.add_legend()


plt.savefig('output images/pair_plot_aware_logx.png', dpi=120)
