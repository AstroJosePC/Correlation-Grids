import copy
import functools
from datetime import datetime
from os.path import isdir, join, isfile

import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import algorithms as algo
from seaborn import utils
from seaborn._decorators import _deprecate_positional_args
from seaborn.regression import _RegressionPlotter

from .utils import linmix as linmix_method


def fill_between(ax, x, y1, y2, *, where=None, interpolate=False, step=None, **kwargs):
    ind_dir = 'x'
    dep_dir = {"x": "y", "y": "x"}[ind_dir]
    func_name = {"x": "fill_between", "y": "fill_betweenx"}[dep_dir]

    if not mpl.rcParams["_internal.classic_mode"]:
        kwargs = cbook.normalize_kwargs(kwargs, mcoll.Collection)
    if not any(c in kwargs for c in ("color", "facecolor")):
        kwargs["facecolor"] = ax._get_patches_for_fill.get_next_color()
    # Handle united data, such as dates
    ax._process_unit_info(
        **{f"{ind_dir}data": x, f"{dep_dir}data": y1}, kwargs=kwargs)
    ax._process_unit_info(
        **{f"{dep_dir}data": y2})

    # Convert the arrays so we can work with them
    x = np.ma.masked_invalid(getattr(ax, f"convert_{ind_dir}units")(x))
    y1 = np.ma.masked_invalid(
        getattr(ax, f"convert_{dep_dir}units")(y1))
    y2 = np.ma.masked_invalid(
        getattr(ax, f"convert_{dep_dir}units")(y2))

    for name, array in [
        (ind_dir, x), (f"{dep_dir}1", y1), (f"{dep_dir}2", y2)]:
        if array.ndim > 1:
            raise ValueError(f"{name!r} is not 1-dimensional")

    if where is None:
        where = True
    else:
        where = np.asarray(where, dtype=bool)
        if where.size != x.size:
            cbook.warn_deprecated(
                "3.2", message=f"Since %(since)s, the parameter *where* "
                               f"must have the same size as {x} in {func_name}(). This "
                               "will become an error %(removal)s.")
    where = where & ~functools.reduce(
        np.logical_or, map(np.ma.getmask, [x, y1, y2]))

    x, y1, y2 = np.broadcast_arrays(np.atleast_1d(x), y1, y2)

    polys = []
    for idx0, idx1 in cbook.contiguous_regions(where):
        indslice = x[idx0:idx1]
        dep1slice = y1[idx0:idx1]
        dep2slice = y2[idx0:idx1]
        if step is not None:
            step_func = cbook.STEP_LOOKUP_MAP["steps-" + step]
            indslice, dep1slice, dep2slice = \
                step_func(indslice, dep1slice, dep2slice)

        if not len(indslice):
            continue

        N = len(indslice)
        pts = np.zeros((2 * N + 2, 2))

        if interpolate:
            def get_interp_point(idx):
                im1 = max(idx - 1, 0)
                ind_values = x[im1:idx + 1]
                diff_values = y1[im1:idx + 1] - y2[im1:idx + 1]
                dep1_values = y1[im1:idx + 1]

                if len(diff_values) == 2:
                    if np.ma.is_masked(diff_values[1]):
                        return x[im1], y1[im1]
                    elif np.ma.is_masked(diff_values[0]):
                        return x[idx], y1[idx]

                diff_order = diff_values.argsort()
                diff_root_ind = np.interp(
                    0, diff_values[diff_order], ind_values[diff_order])
                ind_order = ind_values.argsort()
                diff_root_dep = np.interp(
                    diff_root_ind,
                    ind_values[ind_order], dep1_values[ind_order])
                return diff_root_ind, diff_root_dep

            start = get_interp_point(idx0)
            end = get_interp_point(idx1)
        else:
            # Handle scalar dep2 (e.g. 0): the fill should go all
            # the way down to 0 even if none of the dep1 sample points do.
            start = indslice[0], dep2slice[0]
            end = indslice[-1], dep2slice[-1]

        pts[0] = start
        pts[N + 1] = end

        pts[1:N + 1, 0] = indslice
        pts[1:N + 1, 1] = dep1slice
        pts[N + 2:, 0] = indslice[::-1]
        pts[N + 2:, 1] = dep2slice[::-1]

        if ind_dir == "y":
            pts = pts[:, ::-1]

        polys.append(pts)

    collection = mcoll.PolyCollection(polys, **kwargs)

    # now update the datalim and autoscale
    pts = np.row_stack([np.column_stack([x[where], y1[where]]),
                        np.column_stack([x[where], y2[where]])])
    if ind_dir == "y":
        pts = pts[:, ::-1]
    # ax.update_datalim(pts, updatex=True, updatey=True)
    ax.add_collection(collection, autolim=False)
    # ax._request_autoscale_view()
    return collection

@_deprecate_positional_args
def regplot_log(
        *,
        x=None, y=None, xerr=None, yerr=None,
        data=None,
        x_estimator=None, x_bins=None, x_ci="ci",
        scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None,
        seed=None, order=1, logistic=False, lowess=False, robust=False, linmix=False, linmix_path=None,
        logx=False, logy=False, x_partial=None, y_partial=None, xdelta=None, ydelta=None, linmix_kws=None,
        truncate=False, fit_xrange=None, x_jitter=None, y_jitter=None,
        label=None, color=None, marker="o", size=50, ann_coeff=False,
        scatter_kws=None, line_kws=None, ax=None):
    plotter = _RegressionPlotter_Log(x, y, xerr=xerr, yerr=yerr, data=data, x_estimator=x_estimator, x_bins=x_bins,
                                     x_ci=x_ci,
                                     scatter=scatter, fit_reg=fit_reg, ci=ci, n_boot=n_boot, units=units, seed=seed,
                                     order=order, logistic=logistic, lowess=lowess, robust=robust,
                                     logx=logx, logy=logy, linmix=linmix, linmix_path=linmix_path,
                                     x_partial=x_partial, y_partial=y_partial, xdelta=xdelta, ydelta=ydelta,
                                     truncate=truncate, linmix_kws=linmix_kws,
                                     x_jitter=x_jitter, y_jitter=y_jitter, color=color, label=label)

    if ax is None:
        ax = plt.gca()

    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)

    markers = {'upp': r'$\downarrow$', 'norm': marker}
    sizes = {'upp': 230, 'norm': size}

    scatter_kws["markers"] = markers
    scatter_kws["sizes"] = sizes

    line_kws = {} if line_kws is None else copy.copy(line_kws)
    line_kws.setdefault('x_range', fit_xrange)
    line_kws.setdefault('ann_coeff', ann_coeff)
    plotter.plot(ax, scatter_kws, line_kws)
    ax.set_xlim(fit_xrange[0], fit_xrange[1])
    return ax, plotter


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
        self._chain = None

        # Validate the regression options:
        if sum((order > 1, logistic, robust, lowess, logx)) > 1:
            raise ValueError("Mutually exclusive regression options.")

        # Extract the data vals from the arguments or passed dataframe
        self.establish_variables(data, x=x, y=y, xerr=xerr, yerr=yerr, ydelta=ydelta, units=units,
                                 x_partial=x_partial, y_partial=y_partial)
        # Filter out bad data
        self.filter_out("x", "y", 'xerr', 'yerr', "ydelta", "units", "x_partial", "y_partial",
                        xdelta=xdelta, ydelta=ydelta)

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

    def filter_out(self, *vars, xdelta=None, ydelta=None):
        # Dummy variable to drop invalid upper limits
        drop_upp = np.zeros(self.x.size)

        # Remove x non-detections: only y non-detecitons are allowed!
        if xdelta is not None:
            drop_upp[~xdelta.astype(bool)] = np.nan
        # Remove x errors equal to zero; they make no sense!?
        if self.xerr is not None:
            drop_upp[self.xerr == 0.0] = np.nan

        # Remove x errors equal to zero for detections
        if self.yerr is not None:
            to_drop = self.yerr == 0.0
            if ydelta is not None:
                detect = ydelta.astype(bool)
                to_drop = to_drop & detect
            drop_upp[to_drop] = np.nan

        # Remove y non-detections for methods other than linmix
        if not self.linmix and ydelta is not None:
            drop_upp[~ydelta.astype(bool)] = np.nan

        # Drop null observations, and those marked above
        vals = [getattr(self, var) for var in vars]
        vals = [v for v in vals if v is not None] + [drop_upp]
        not_na = np.all(np.column_stack([pd.notnull(v) for v in vals]), axis=1)
        for var in vars:
            val = getattr(self, var)
            if val is not None:
                setattr(self, var, val[not_na])

    def fit_regression(self, ax=None, x_range=None, grid=None):
        """Fit the regression model."""
        # Create the grid for the regression
        if grid is None:
            if self.truncate:
                x_min, x_max = self.x_range
            elif x_range is not None:
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
                xerr = (1 / np.log(10)) * self.xerr / self.x
            else:
                xerr = self.xerr
        else:
            x = self.x
            xerr = self.xerr
        if self.logy:
            y = np.log10(self.y)
            if self.yerr is not None:
                yerr = (1 / np.log(10)) * self.yerr / self.y
            else:
                yerr = self.yerr
        else:
            y = self.y
            yerr = self.yerr

        lm = linmix_method.LinMix(x, y, xsig=xerr, ysig=yerr, delta=self.ydelta, **self.linmix_kws)
        lm.run_mcmc(maxiter=self.n_boot, silent=True)
        self._chain = lm.chain
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
            np.seterr(over='ignore')
            yhat_boots = 10 ** yhat_boots
            np.seterr(over='warn')
            yhat = 10 ** yhat
        return yhat, yhat_boots

    def scatterplot(self, ax, kws):
        """Draw the data."""
        # Treat the line-based markers specially, explicitly setting larger
        # linewidth than is provided by the seaborn style defaults.
        # This would ideally be handled better in matplotlib (i.e., distinguish
        # between edgewidth for solid glyphs and linewidth for line glyphs
        # but this should do for now.
        line_markers = ["1", "2", "3", "4", "+", "x", "|", "_"]
        if self.x_estimator is None:
            if "markers" in kws and kws["markers"]['norm'] in line_markers:
                lw = mpl.rcParams["lines.linewidth"]
            else:
                lw = mpl.rcParams["lines.markeredgewidth"]
            kws.setdefault("linewidths", lw)
            kws.setdefault("alpha", .8)

            x, y = self.scatter_data
            if self.ydelta is not None and np.any(self.ydelta == 0):
                # If we got upper limits, then draw using seaborn
                detections = ['norm' if detection else 'upp' for detection in self.ydelta]
                sns.scatterplot(x=x, y=y, style=detections, size=detections, legend=False, ax=ax, **kws)
            else:
                if "markers" in kws:
                    markers = kws.pop('markers')
                    kws.setdefault('marker', markers['norm'])
                if 'sizes' in kws:
                    sizes = kws.pop('sizes')
                    kws.setdefault('s', sizes['norm'])
                ax.scatter(x, y, **kws)
        else:
            # TODO abstraction
            ci_kws = {"color": kws["color"]}
            ci_kws["linewidth"] = mpl.rcParams["lines.linewidth"] * 1.75
            kws.setdefault("s", 50)

            xs, ys, cis = self.estimate_data
            if [ci for ci in cis if ci is not None]:
                for x, ci in zip(xs, cis):
                    ax.plot([x, x], ci, **ci_kws)
            ax.scatter(xs, ys, **kws)

    def lineplot(self, ax, kws):
        """Draw the model."""
        ann_coeff = kws.pop('ann_coeff', False)
        # Fit the regression model
        grid, yhat, err_bands = self.fit_regression(ax, x_range=kws.pop('x_range', None))
        edges = grid[0], grid[-1]

        # Get set default aesthetics
        fill_color = kws["color"]
        lw = kws.pop("lw", mpl.rcParams["lines.linewidth"] * 1.5)
        kws.setdefault("linewidth", lw)

        # Draw the regression line and confidence interval
        line, = ax.plot(grid, yhat, **kws)
        line.sticky_edges.x[:] = edges  # Prevent mpl from adding margin
        if err_bands is not None:
            fill_between(ax, grid, *err_bands, facecolor=fill_color, alpha=.15)
        if ann_coeff:
            self._ann_coeff(ax)

    def _ann_coeff(self, ax):
        if self.linmix and self._chain is not None:
            corr_coeff = np.median(self._chain['corr'])
        else:
            import scipy.stats as stats
            corr_coeff, _ = stats.pearsonr(self.x, self.y)

        ax.text(s=f' coeff: {corr_coeff:.2f}', x=0.05, y=0.95, transform=ax.transAxes,
                bbox={'boxstyle': 'round', 'pad': 0.25, 'facecolor': 'white', 'edgecolor': 'gray'})


if __name__ == '__main__':
    visir = pd.read_csv('Data/VISIR_merged_fluxes_TMP.csv', sep=',',
                        skipinitialspace=True, na_values=['#NAME?'])

    visir['flux_x_corr'] = visir['flux_x'].copy()
    upp_mask = visir['fl_err_x'] > visir['flux_x']
    ydelta = (~upp_mask).astype(int)
    visir.loc[upp_mask, 'flux_x_corr'] = 2 * visir['fl_err_x'][upp_mask]

    fig1, (ax11, ax12) = plt.subplots(nrows=2, constrained_layout=True, )

    regplot_log(data=visir, x='Mstar', y='fwhm_x', ax=ax11)
    ax11.set_title('regular fit')

    regplot_log(data=visir, x='Mstar', y='fwhm_x', linmix=True, ax=ax12, linmix_path='../test_regplot')
    ax12.set_title('linmix fit')
    fig1.suptitle('regular vs linmix fit (linear scales)')
    plt.show()

    fig2, (ax21, ax22) = plt.subplots(nrows=2, constrained_layout=True, )
    regplot_log(data=visir, x='Mstar', y='fwhm_x', logx=True, logy=True, n_boot=10000, ax=ax21)
    ax21.set_title('regular fit')
    ax21.set(yscale='log', xscale='log')
    regplot_log(data=visir, x='Mstar', y='fwhm_x', logx=True, logy=True, linmix=True,
                n_boot=10000, ax=ax22, linmix_path='../test_regplot')
    ax22.set(yscale='log', xscale='log')
    ax22.set_title('linmix fit')
    fig2.suptitle('regular vs linmix fit (log scales)')
    plt.show()

    fig3, (ax31, ax32) = plt.subplots(nrows=2, constrained_layout=True)
    regplot_log(data=visir, x='Mstar', y='flux_x_corr', yerr='fl_err_x', ydelta=ydelta,
                logx=True, logy=True, n_boot=10000, ax=ax31)
    ax31.set_title('regular fit (no upper limits used)')
    ax31.set(yscale='log', xscale='log')
    regplot_log(data=visir, x='Mstar', y='flux_x_corr', yerr='fl_err_x', ydelta=ydelta,
                logx=True, logy=True, linmix=True,
                n_boot=10000, ax=ax32, linmix_path='../test_regplot')
    ax32.set(yscale='log', xscale='log')
    ax32.set_title('linmix fit')
    fig3.suptitle('regular vs linmix fit (upper limits)')
    plt.show()
