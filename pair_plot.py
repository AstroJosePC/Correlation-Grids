import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme()
sns.set_context("paper")


# Function to calculate correlation coefficient between two arrays
def corr(x, y, **kwargs):
    # https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
    # Calculate the value
    coeff = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coeff, 2))

    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)


spitzer = pd.read_csv('Data/Spitzer_ALMA_sample.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])

spitzer['Rco / Rsubl'] = spitzer['CORAD'] / spitzer['RSUBL']
spitzer['Rco / Rsnow'] = spitzer['CORAD'] / spitzer['RSNOW_ACC']
spitzer['R68%'] = np.log10(spitzer['LOGRDUST68'])
spitzer['Rdust'] = np.log10(spitzer['LOGRDUST95'])

variables = ['MSTAR', 'LSTAR', 'LOGLACC', 'LOGMACC', 'TEFF', 'N1331', 'FNIR', 'DIST', 'INCL', 'F09MM',
             'F13MM', 'R68%', 'Rdust', 'COV2V1', 'CORAD', 'RSUBL', 'Rco / Rsubl', 'Rco / Rsnow',
             'FLCO_BC', 'FLCO_NC', 'FLH2O_17', 'FLHCN', 'FLC2H2', 'FLCO2', 'CONT_17', 'FLH2_17']

# variables = ['MSTAR', 'LSTAR', 'LOGLACC', 'LOGMACC', 'TEFF', 'N1331', 'DIST', 'INCL']

g = sns.PairGrid(spitzer, dropna=True, vars=variables, diag_sharey=False, corner=True)
g.map_lower(sns.regplot)
g.map_lower(corr)
g.map_diag(sns.histplot)
# f.savefig('pair_plot2.png', dpi=150)

plt.savefig('output images/pair_plot2.png', dpi=100)

# plt.show()
# g.map_diag(sns.histplot, hue=None, color=".3")
# g.map_offdiag(sns.scatterplot)
# g.add_legend()
