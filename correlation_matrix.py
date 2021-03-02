import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


spitzer = pd.read_csv('Data/Spitzer_ALMA_sample.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])

spitzer['Rco / Rsubl'] = spitzer['CORAD'] / spitzer['RSUBL']
spitzer['Rco / Rsnow'] = spitzer['CORAD'] / spitzer['RSNOW_ACC']
spitzer['R68%'] = np.log10(spitzer['LOGRDUST68'])
spitzer['Rdust'] = np.log10(spitzer['LOGRDUST95'])


variables = ['MSTAR', 'LSTAR', 'LOGLACC', 'LOGMACC', 'TEFF',
             'N1331', 'FNIR', 'DIST', 'INCL', 'F09MM', 'F13MM', 'R68%', 'Rdust', 'COV2V1', 'CORAD', 'RSUBL', 'Rco / Rsubl',
             'Rco / Rsnow', 'FLCO_BC', 'FLCO_NC', 'FLH2O_17', 'FLHCN', 'FLC2H2', 'FLCO2', 'CONT_17', 'FLH2_17']

# Compute the correlation matrix
corr = spitzer[variables].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0, fmt='.1g',
                square=True, linewidths=1.5, cbar_kws={"shrink": 0.5}, annot=True, cbar=False)


# g.set_xticklabels(rotation=30)
# g.set_xticklabels(ax.get_xticklabels(), fontsize=7)
# plt.tight_layout()

plt.show()
