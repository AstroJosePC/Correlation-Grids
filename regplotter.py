import copy
from os.path import isdir, join, isfile
from datetime import datetime
import numpy as np
from seaborn import utils
from seaborn.regression import _RegressionPlotter
from seaborn._decorators import _deprecate_positional_args
from seaborn import algorithms as algo
from utils import linmix
import matplotlib.pyplot as plt


@_deprecate_positional_args
def regplot_log(
        *,
        x=None, y=None, xerr=None, yerr=None,
        data=None,
        x_estimator=None, x_bins=None, x_ci="ci",
        scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None,
        seed=None, order=1, logistic=False, lowess=False, robust=False, linmix=False, linmix_path=None,
        logx=False, logy=False, x_partial=None, y_partial=None, xdelta=None, ydelta=None, linmix_kws=None,
        truncate=True, x_jitter=None, y_jitter=None,
        label=None, color=None, marker="o",
        scatter_kws=None, line_kws=None, ax=None):
    plotter = _RegressionPlotter_Log(x, y, xerr=xerr, yerr=yerr, data=data, x_estimator=x_estimator, x_bins=x_bins, x_ci=x_ci,
                                     scatter=scatter, fit_reg=fit_reg, ci=ci, n_boot=n_boot, units=units, seed=seed,
                                     order=order, logistic=logistic, lowess=lowess, robust=robust,
                                     logx=logx, logy=logy, linmix=linmix, linmix_path=linmix_path,
                                     x_partial=x_partial, y_partial=y_partial, xdelta=xdelta, ydelta=ydelta,
                                     truncate=truncate, linmix_kws=linmix_kws,
                                     x_jitter=x_jitter, y_jitter=y_jitter, color=color, label=label)

    if ax is None:
        ax = plt.gca()

    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
    scatter_kws["marker"] = marker
    line_kws = {} if line_kws is None else copy.copy(line_kws)
    plotter.plot(ax, scatter_kws, line_kws)
    return ax


class _RegressionPlotter_Log(_RegressionPlotter):
    """
    Plotter for numeric independent variables with regression model.
    This does the computations and drawing for the `regplot_log` function, and
    is thus also used indirectly by `lmplot`.
    """

    def __init__(self, x, y, data=None, xerr=None, yerr=None, x_estimator=None, x_bins=None,
                 x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=1000,
                 units=None, seed=None, order=1, logistic=False, lowess=False,
                 robust=False, logx=False, logy=False, linmix=False,
                 x_partial=None, y_partial=None, xdelta=None, ydelta=None,
                 truncate=False, x_jitter=None, y_jitter=None,
                 color=None, label=None, linmix_path=None, linmix_kws=None):

        # Set member attributes
        self.x_estimator = x_estimator
        self.ci = ci
        self.x_ci = ci if x_ci == "ci" else x_ci
        self.n_boot = n_boot
        self.seed = seed
        self.scatter = scatter
        self.fit_reg = fit_reg
        self.order = order
        self.logistic = logistic
        self.lowess = lowess
        self.robust = robust
        self.logx = logx
        self.logy = logy
        self.linmix = linmix
        self.linmix_kws = linmix_kws if linmix_kws is not None else {}
        self.truncate = truncate
        self.x_jitter = x_jitter
        self.y_jitter = y_jitter
        self.color = color
        self.label = label

        # Validate the regression options:
        if sum((order > 1, logistic, robust, lowess, logx)) > 1:
            raise ValueError("Mutually exclusive regression options.")

        # Extract the data vals from the arguments or passed dataframe
        self.establish_variables(data, x=x, y=y, xerr=xerr, yerr=yerr, ydelta=ydelta, units=units,
                                 x_partial=x_partial, y_partial=y_partial)

        # Mark x upper limits as null values; only y upper limits are allowed
        # dummy variable to drop invalid upper limits
        self._drop_upp = np.zeros(self.x.size)
        if xdelta is not None:
            self._drop_upp[~self.xdelta.astype(bool)] = np.nan
        # Mark y upper limits for any method other than linmix as null values; only linmix can use upper limits
        if ydelta is not None and not linmix:
            self._drop_upp[~self.ydelta.astype(bool)] = np.nan
        # Drop null observations
        self.dropna("x", "y", 'xerr', 'yerr', "ydelta", "units", "x_partial", "y_partial", "_drop_upp")

        # Regress nuisance variables out of the data
        if self.x_partial is not None:
            self.x = self.regress_out(self.x, self.x_partial)
        if self.y_partial is not None:
            self.y = self.regress_out(self.y, self.y_partial)

        # Possibly bin the predictor variable, which implies a point estimate
        if x_bins is not None:
            self.x_estimator = np.mean if x_estimator is None else x_estimator
            x_discrete, x_bins = self.bin_predictor(x_bins)
            self.x_discrete = x_discrete
        else:
            self.x_discrete = self.x

        # Disable regression in case of singleton inputs
        if len(self.x) <= 1:
            self.fit_reg = False

        # Save the range of the x variable for the grid later
        if self.fit_reg:
            self.x_range = self.x.min(), self.x.max()

        # Check if path to linmix output is a proper directory
        self.linmix_path = linmix_path
        if self.linmix_path is not None:
            self.save_linmix = isdir(linmix_path)
            if not self.save_linmix:
                raise ValueError(f'The following directory to save linmix output does not exist: {linmix_path} ')
        else:
            self.save_linmix = False

    def fit_regression(self, ax=None, x_range=None, grid=None):
        """Fit the regression model."""
        # Create the grid for the regression
        if grid is None:
            if self.truncate:
                x_min, x_max = self.x_range
            else:
                if ax is None:
                    x_min, x_max = x_range
                else:
                    x_min, x_max = ax.get_xlim()
            grid = np.linspace(x_min, x_max, 100)
        ci = self.ci

        # Fit the regression
        if self.order > 1:
            yhat, yhat_boots = self.fit_poly(grid, self.order)
        elif self.linmix:
            yhat, yhat_boots = self.fit_linmix(grid)
        elif self.logistic:
            from statsmodels.genmod.generalized_linear_model import GLM
            from statsmodels.genmod.families import Binomial
            yhat, yhat_boots = self.fit_statsmodels(grid, GLM,
                                                    family=Binomial())
        elif self.lowess:
            ci = None
            grid, yhat = self.fit_lowess()
        elif self.robust:
            from statsmodels.robust.robust_linear_model import RLM
            yhat, yhat_boots = self.fit_statsmodels(grid, RLM)
        elif self.logx and self.logy:
            yhat, yhat_boots = self.fit_logxy(grid)
        elif self.logy:
            yhat, yhat_boots = self.fit_logy(grid)
        elif self.logx:
            yhat, yhat_boots = self.fit_logx(grid)
        else:
            yhat, yhat_boots = self.fit_fast(grid)

        # Compute the confidence interval at each grid point
        if ci is None:
            err_bands = None
        else:
            err_bands = utils.ci(yhat_boots, ci, axis=0)

        return grid, yhat, err_bands

    def fit_logxy(self, grid):
        """Fit the model in log-xy-space."""
        X, y = np.c_[np.ones(len(self.x)), self.x], self.y
        grid = np.c_[np.ones(len(grid)), np.log(grid)]

        def reg_func(_x, _y):
            _x = np.c_[_x[:, 0], np.log(_x[:, 1])]
            _y = np.log10(_y)
            return np.linalg.pinv(_x).dot(_y)

        yhat = grid.dot(reg_func(X, y))
        if self.ci is None:
            return yhat, None

        beta_boots = algo.bootstrap(X, y,
                                    func=reg_func,
                                    n_boot=self.n_boot,
                                    units=self.units,
                                    seed=self.seed).T
        yhat_boots = grid.dot(beta_boots).T
        return 10 ** yhat, 10 ** yhat_boots

    def fit_logy(self, grid):
        """Fit the model in logy-space."""

        def reg_func(_x, _y):
            _y = np.log10(_y)
            return np.linalg.pinv(_x).dot(_y)

        X, y = np.c_[np.ones(len(self.x)), self.x], self.y
        grid = np.c_[np.ones(len(grid)), grid]
        yhat = grid.dot(reg_func(X, y))
        if self.ci is None:
            return yhat, None

        beta_boots = algo.bootstrap(X, y,
                                    func=reg_func,
                                    n_boot=self.n_boot,
                                    units=self.units,
                                    seed=self.seed).T
        yhat_boots = grid.dot(beta_boots).T
        return 10 ** yhat, 10 ** yhat_boots

    def fit_linmix(self, grid):
        """Fit the model using linmix and save output to file (optional)"""
        if self.logx:
            x = np.log10(self.x)
            if self.xerr is not None:
                xerr = (1/np.log(10)) * self.xerr / x
            else:
                xerr = self.xerr
        else:
            x = self.x
            xerr = self.xerr
        if self.logy:
            y = np.log10(self.y)
            if self.yerr is not None:
                yerr = (1/np.log(10)) * self.yerr / y
            else:
                yerr = self.yerr
        else:
            y = self.y
            yerr = self.yerr

        lm = linmix.LinMix(x, y, xsig=xerr, ysig=yerr, delta=self.ydelta, **self.linmix_kws)
        print('created LinMix object')
        print('Starting MCMC')
        lm.run_mcmc(maxiter=self.n_boot, silent=True)
        print('Finished MCMC')
        # Compute median values, standard deviations of results
        beta_median = np.median(lm.chain['beta'])
        alpha_median = np.median(lm.chain['alpha'])

        # Predict values on grid (from median regression values)
        if self.logx:
            grid = np.log10(grid)
        yhat = alpha_median + beta_median * grid
        yhat_boots = lm.chain['alpha'][:, None] + grid * lm.chain['beta'][:, None]

        if self.save_linmix:
            cols = ['alpha', 'beta', 'sigsqr', 'mu0', 'usqr', 'wsqr', 'ximean', 'xisig', 'corr']
            x_name = self.x.name if hasattr(self.x, "name") else ''
            y_name = self.y.name if hasattr(self.y, "name") else ''
            if x_name and y_name:
                basename = '_'.join([y_name, x_name, 'linmix'])
            else:
                basename = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + '_linmix'
            filename = basename + '.csv'
            filepath = join(self.linmix_path, filename)
            if isfile(filepath):
                print(f'WARNING: Overwriting the following file: {filepath}')
            pd.DataFrame.from_records(lm.chain[cols]).to_csv(filepath, index=False)
        if self.logy:
            return 10 ** yhat, 10 ** yhat_boots
        else:
            return yhat, yhat_boots


if __name__ == '__main__':
    import pandas as pd

    visir = pd.read_csv('Data\VISIR_merged_fluxes_TMP.csv',
                        sep=',', skipinitialspace=True, na_values=['#NAME?'])

    fig1, (ax11, ax12) = plt.subplots(nrows=2, constrained_layout=True, sharex='col')

    regplot_log(data=visir, x='Mstar', y='fwhm_x', ax=ax11)
    ax11.set_title('regular fit')

    regplot_log(data=visir, x='Mstar', y='fwhm_x', linmix=True, ax=ax12, linmix_path='.\\test')
    ax12.set_title('linmix fit')
    plt.show()

    fig2, (ax21, ax22) = plt.subplots(nrows=2, constrained_layout=True, sharex='col')
    regplot_log(data=visir, x='Mstar', y='fwhm_x', logx=True, logy=True, n_boot=10000, ax=ax21)
    ax21.set_title('regular fit')
    ax21.set(yscale='log', xscale='log')
    regplot_log(data=visir, x='Mstar', y='fwhm_x',  logx=True, logy=True, linmix=True,
                n_boot=10000, ax=ax22, linmix_path='./Test')
    ax22.set(yscale='log', xscale='log')
    ax22.set_title('linmix fit')
    plt.show()
