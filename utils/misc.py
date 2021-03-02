import matplotlib.pyplot as plt


def is_log(x, y, log_vars=None, **kwargs):
    # Map log scales
    # pass
    # Get current axis
    ax = plt.gca()

    if x.name in log_vars:
        ax.set_xscale('log')
    if y.name in log_vars:
        ax.set_yscale('log')
    ax.autoscale()