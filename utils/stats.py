import numpy as np
import matplotlib.pyplot as plt


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
    # x_vars, y_vars = get_vars(x, y, log_vars)
    # coeff = np.corrcoef(x_vars, y_vars)[0][1]
    # # Make the label
    # label = r'$\rho$ = ' + str(round(coeff, 2))
    #
    # # Add the label to the plot
    # ax = plt.gca()
    # ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)
    pass