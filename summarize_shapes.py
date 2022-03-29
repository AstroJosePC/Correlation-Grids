import os

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter

from corr_grid import flux_prep, parse_err_map
from tools import remove_disks, wind_disk_design
from utils import linmix

markers = {'west': r'$\leftarrow$', 'southwest': r'$\swarrow$', 'south': r'$\downarrow$', 'norm': 'o'}
sizes = {'west': 230, 'southwest': 230, 'south': 230, 'norm': 50}

# REMOVE FOLLOWING TARGETS:
low_mass = ['WaOph6', 'AA Tau']
outliers = ['MWC297', 'TW Hya']
fw10_outlier = ['IQ Tau']

# Other params
log_vars = ['MSTAR', 'LSTAR', 'TEFF', 'V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX']
# plotter_wrap = plotter_wrapper(regplot_log_wrap, labels=LABELS, stats=STATS)
labels_map = {'V1_FLUX': r'CO v1-0 Lum (L$_\odot$)',
              'V1H_FLUX': r'CO v1H-0H Lum (L$_\odot$)',
              'V2_FLUX': r'CO v2-1 Lum (L$_\odot$)',
              'I13_FLUX': r'$^{13}$CO v1-0 Lum (L$_\odot$)'}


# Useful function:
def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)


def _transform_data(z, zerr=None, forward: bool = True, logz: bool = True):
    """
    Transform data from linear from or to log scale. Forward boolean input determines whether we covnert to (or from) log scale.
    """
    if logz:
        if forward:
            if zerr is not None:
                zerr = (1 / np.log(10)) * zerr / z
            newz = np.log10(z)
        else:
            newz = 10 ** z
            if zerr is not None:
                zerr = zerr * z * np.log(10)
    else:
        newz = z
    # return the new transformed values (yes these zerr may be different)
    return newz, zerr


def retrieve_chains(x, y, xsig, ysig, delta, mask):
    xsig = xsig[mask] if xsig is not None else None
    ysig = ysig[mask] if ysig is not None else None
    delta = delta[mask] if delta is not None else None

    linmix_obj = linmix.LinMix(x[mask], y[mask], xsig=xsig, ysig=ysig,
                               delta=delta, seed=31)
    linmix_obj.run_mcmc(maxiter=10000, silent=True)
    return linmix_obj.chain


def clean_data(data, *args):
    return data.dropna(subset=[var for var in args if var is not None]).copy()


def z_to_plot(z, logz):
    """
    convert data originally in log scale if needed
    """
    return 10 ** z if logz else z


def main(force=False):
    # FOR iSHELL + CRIRES Dataset
    dataset = pd.read_csv('Data/shapes dataset v2.csv', sep=',', skipinitialspace=True, na_values=['#NAME?'])
    dataset = remove_disks(dataset, outliers + low_mass)

    xvars = ['MSTAR', 'LSTAR', 'LOGLACC', 'INCL', 'N1330', 'TEFF', 'FW10']
    yvars = ['V1_FLUX', 'V1H_FLUX', 'V2_FLUX', 'I13_FLUX', 'FW10']

    delta_map = flux_prep(dataset, y_vars=yvars, log_vars=log_vars, labels_map=labels_map, correction_col='W2_JY')
    markers, hue_order, linestyles, pass_subsets = wind_disk_design()
    colors = sns.color_palette('colorblind', 2)
    sizes = {r' ': 70, 's': 70, '^': 70, r'$\downarrow$': 230}

    error_map = parse_err_map(dataset, col_set=xvars + yvars)

    for yvar in yvars:
        logy = yvar in log_vars
        yvar_err = error_map.get(yvar)
        delta_key = delta_map.get(yvar)
        markers_key = yvar + '_markers'
        sizes_key = yvar + '_sizes'

        for xvar in xvars:
            path = rf'output images/summaries/{yvar} vs {xvar}.png'
            if (os.path.isfile(path) and not force) or xvar == yvar:
                continue

            logx = xvar in log_vars
            xvar_err = error_map.get(xvar)

            clean_dataset = clean_data(dataset, xvar, xvar_err, yvar, yvar_err, delta_key)

            # sort out markers & sizes
            clean_dataset[markers_key] = r' '

            det_mask = clean_dataset[delta_key].astype(bool) if delta_key else np.ones(len(clean_dataset), dtype=bool)
            winds_mask = clean_dataset['shape'] == 'winds'
            disk_mask = clean_dataset['shape'] == 'disk'

            clean_dataset.loc[winds_mask & det_mask, markers_key] = r'^'
            clean_dataset.loc[disk_mask & det_mask, markers_key] = r's'
            clean_dataset.loc[~det_mask, markers_key] = r'$\downarrow$'

            clean_dataset[sizes_key] = 150
            clean_dataset.loc[~det_mask, sizes_key] = 230

            x, xerr = clean_dataset[xvar], clean_dataset[xvar_err] if xvar_err else None
            y, yerr = clean_dataset[yvar], clean_dataset[yvar_err] if yvar_err else None

            x, xerr = _transform_data(x, xerr, logz=logx)
            y, yerr = _transform_data(y, yerr, logz=logy)

            fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(nrows=2, ncols=2, figsize=(10, 6),
                                                         constrained_layout=True, )
            sns.scatterplot(x=xvar, y=yvar, data=clean_dataset, hue='shape', legend=False, style=markers_key,
                            markers=markers if np.all(det_mask) else markers + [r'$\downarrow$'],
                            size=markers_key, sizes=sizes, alpha=0.8, ax=ax1)

            bounds = dict(sigma=[], beta=[], corr=[])

            for i, shape, subset_mask in zip([0, 1], ['winds', 'disks'], [winds_mask, disk_mask]):
                print(f'Fitting linmix for {xvar} and {yvar}: {shape}')
                chain = retrieve_chains(x, y, xerr, yerr, clean_dataset[delta_key] if delta_key else np.ones(len(clean_dataset)),
                                        subset_mask)

                median_beta = chain['beta']
                median_alpha = chain['alpha']
                reg_x = np.array([x[subset_mask].min(), x[subset_mask].max()])
                reg_y = np.median(median_alpha) + np.median(median_beta) * reg_x

                ax1.plot(10 ** reg_x if logx else reg_x,
                         10 ** reg_y if logy else reg_y,
                         '--', c=colors[i])

                hdi_reg_x = np.linspace(*reg_x, 100)
                reg_lines = chain['beta'][:, None] * hdi_reg_x + chain['alpha'][:, None]

                # noinspection PyTypeChecker
                lines_hdi = arviz.hdi(reg_lines[np.newaxis, ...])

                top_line = savgol_filter(lines_hdi[:, 1], 17, 3)
                bot_line = savgol_filter(lines_hdi[:, 0], 17, 3)

                ax1.fill_between(10 ** hdi_reg_x if logx else hdi_reg_x,
                                 10 ** bot_line if logy else bot_line,
                                 10 ** top_line if logy else top_line,
                                 color=colors[i],
                                 alpha=0.2)

                sigma_bounds = ci(chain['sigsqr'], axis=0, which=89)
                beta_bounds = ci(chain['beta'], axis=0, which=95)
                corr_bounds = ci(chain['corr'], axis=0, which=98)

                bounds['sigma'].extend(sigma_bounds)
                bounds['beta'].extend(beta_bounds)
                bounds['corr'].extend(corr_bounds)

                corr_hdi = arviz.hdi(chain['corr'])
                beta_hdi = arviz.hdi(chain['beta'])
                sigma_hdi = arviz.hdi(chain['sigsqr'])

                # Histogram stuff
                sns.histplot(x=chain['corr'], ax=ax2, stat="count", common_norm=False, color=colors[i], fill=False,
                             element="step")

                corr_hist, corr_edges = sns._statistics.Histogram(stat='count')(chain['corr'])
                corr_hdi_mask = (corr_hdi[0] <= corr_edges[:-1]) & (corr_edges[1:] <= corr_hdi[1])
                ax2.fill_between(corr_edges[1:][corr_hdi_mask], 0, corr_hist[corr_hdi_mask], color=colors[i],
                                 alpha=0.2, interpolate=False, step='pre')

                sns.histplot(x=chain['beta'], ax=ax3, stat="count", common_norm=False, color=colors[i], fill=False,
                             element="step")

                beta_hist, beta_edges = sns._statistics.Histogram(stat='count')(chain['beta'])
                beta_hdi_mask = (beta_hdi[0] <= beta_edges[:-1]) & (beta_edges[1:] <= beta_hdi[1])
                ax3.fill_between(beta_edges[1:][beta_hdi_mask], 0, beta_hist[beta_hdi_mask], color=colors[i],
                                 alpha=0.2, interpolate=False, step='pre')

                sns.histplot(x=chain['sigsqr'], ax=ax4, stat="count", common_norm=False, color=colors[i], fill=False,
                             element="step")

                sigma_hist, sigma_edges = sns._statistics.Histogram(stat='count')(chain['sigsqr'])
                sigma_hdi_mask = (sigma_hdi[0] <= sigma_edges[:-1]) & (sigma_edges[1:] <= sigma_hdi[1])
                ax4.fill_between(sigma_edges[1:][sigma_hdi_mask], 0, sigma_hist[sigma_hdi_mask], color=colors[i],
                                 alpha=0.2, interpolate=False, step='pre')

            if logy:
                ax1.set_yscale('log')
            if logx:
                ax1.set_xscale('log')

            beta_bounds = min(*bounds['beta']), max(bounds['beta'])
            sigma_bounds = min(*bounds['sigma']), max(bounds['sigma'])
            corr_bounds = min(*bounds['corr']), max(bounds['corr'])

            ax2.set_xlim(corr_bounds[0] if corr_bounds[0] > -0.98 else 1,
                         corr_bounds[1] if corr_bounds[1] < 0.98 else 1)
            ax2.set(yticklabels=[], yticks=[])

            ax2.set_xlabel(r'Correlation, $\rho$')
            ax2.set_ylabel('')

            ax3.set_xlim(*beta_bounds)
            ax3.set(yticklabels=[], yticks=[])

            ax3.set_xlabel(r'Slope, $\beta$')
            ax3.set_ylabel('')

            ax4.set_xlim(0, sigma_bounds[1])
            ax4.set(yticklabels=[], yticks=[])

            ax4.set_xlabel(r'Instrinsic Scatter Variance, $\sigma^2$')
            ax4.set_ylabel('')

            fig.savefig(path, dpi=200)


if __name__ == '__main__':
    main(False)
