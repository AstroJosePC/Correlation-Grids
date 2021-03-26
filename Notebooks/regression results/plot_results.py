import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

markers = {'west': r'$\leftarrow$', 'southwest': r'$\swarrow$', 'south': r'$\downarrow$', 'norm': 'o'}
sizes = {'west': 230, 'southwest': 230, 'south': 230, 'norm': 50}


# Useful function:
def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)


def output_results(disk_property, lum, lm_chain, upp_mask, property_name, flux_name: str):
    """
    :param disk_property: disk property values
    :param lum: luminosity values
    :param lm_chain: chain output from linmix for current regression
    :param upp_mask: mask for upper limits in lum array
    :param property_name: string name of the property
    :param flux_name: string name of flux line molecule measured
    """

    # Change flux name to just molecule name
    flux_name = flux_name.lstrip('FL')

    limits = ['south' if upp else 'norm' for upp in upp_mask]
    heights = (2, 1.2)
    fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(nrows=2, ncols=2, sharey='row', figsize=(10, 6),
                                                 constrained_layout=True,
                                                 gridspec_kw=dict(height_ratios=heights))

    sns.scatterplot(x=disk_property, y=np.log10(lum), style=limits, hue=limits,
                    markers=markers, size=limits, sizes=sizes, legend=False, ax=ax1)

    xlim = ax1.get_xlim()
    truex = np.array(xlim)
    samplex = truex

    # 75 lines of random samples from alpha and beta
    samples = np.random.choice(lm_chain, size=75)
    for sample in samples:
        sampley = sample['alpha'] + sample['beta'] * samplex
        ax1.plot(samplex, sampley, alpha=0.2)

    sampley = np.median(lm_chain['alpha']) + np.median(lm_chain['beta']) * samplex
    ax1.set_ylabel(flux_name + ' (Lsun)')

    # plot scatter points on ax2
    sns.scatterplot(x=disk_property, y=np.log10(lum), hue=limits, size=limits, style=limits,
                    markers=markers, sizes=sizes, legend=False, ax=ax2)

    # Plot median lines for both ax1, and ax2
    ax1.plot(samplex, sampley, c='red', label='median')
    ax2.plot(samplex, sampley, c='r', alpha=0.9, label='median')

    # 95 percentile bounds; lower and higher
    large_samplex = np.linspace(*samplex, num=10)
    all_lines = lm_chain['alpha'][:, None] + large_samplex * lm_chain['beta'][:, None]
    bounds = ci(all_lines, axis=0)
    ax2.fill_between(large_samplex, *bounds, facecolor='blue', alpha=.15, label='95% CR')

    for ax in [ax1, ax2]:
        ax.set_xlabel(property_name)
        ax.legend()

    # Histogram stuff
    sns.histplot(x=lm_chain['alpha'], ax=ax3, bins=30)
    ax3.set_xlabel(r'$\alpha$')

    alpha_median = np.median(lm_chain['alpha'])
    alpha_mean = np.mean(lm_chain['alpha'])
    ax3.axvline(alpha_median, color='red', label=f"Median")
    ax3.axvline(alpha_mean, color='purple', label=f"Mean")

    sns.histplot(x=lm_chain['beta'], ax=ax4, bins=30)
    ax4.set_xlabel(r'$\beta$')

    beta_median = np.median(lm_chain['beta'])
    beta_mean = np.mean(lm_chain['beta'])

    ax4.axvline(beta_median, color='red', label=f"Median")
    ax4.axvline(beta_mean, color='purple', label=f"Mean")

    alpha_bounds = ci(lm_chain['alpha'], axis=0)
    beta_bounds = ci(lm_chain['beta'], axis=0)

    ax3.fill_between(alpha_bounds, 0, 1, color='green', alpha=0.2,
                     transform=ax3.get_xaxis_transform(), label='95% CR')

    # ax3.text(alpha_bounds[0] - .06, 500, f'{alpha_bounds[0]:.2f}')
    # ax3.text(alpha_bounds[1] + .01, 500, f'{alpha_bounds[1]:.2f}')

    ax4.fill_between(beta_bounds, 0, 1, color='green', alpha=0.2,
                     transform=ax4.get_xaxis_transform(), label='95% CR')
    # ax4.text(beta_bounds[0] - .06, 500, f'{beta_bounds[0]:.2f}')
    # ax4.text(beta_bounds[1] + .01, 500, f'{beta_bounds[1]:.2f}')

    # Median / Mean text
    _, max_ylim = ax3.get_ylim()
    min_xlim, max_xlim = ax3.get_xlim()
    text_xpos = (max_xlim - min_xlim) * 0.2 + min_xlim
    alpha_std = np.std(lm_chain['alpha'])
    ax3.text(text_xpos, max_ylim * 0.9, f'Mean: {alpha_mean:.2f} ± {alpha_std:.2f}')

    _, max_ylim = ax4.get_ylim()
    beta_std = np.std(lm_chain['beta'])
    min_xlim, max_xlim = ax4.get_xlim()
    text_xpos = (max_xlim - min_xlim) * 0.2 + min_xlim
    ax4.text(text_xpos, max_ylim * 0.9, f'Mean: {beta_mean:.2f} ± {beta_std:.2f}')

    for ax in [ax3, ax4]:
        ax.legend()

    title = f"{flux_name} vs {property_name} — coeff: {lm_chain['corr'].mean():.2f} ± {lm_chain['corr'].std():.2f}"
    fig.suptitle(title, fontsize=16)

    fig.savefig(f'results_images/{flux_name}_{property_name}_linmix.jpg', dpi=200, transparent=False)
    plt.close()

    cols = ['alpha', 'beta', 'sigsqr', 'mu0', 'usqr', 'wsqr', 'ximean', 'xisig', 'corr']
    pd.DataFrame.from_records(lm_chain[cols]).to_csv(f'results_csv/{flux_name}_{property_name}_linmix.csv', index=False)
