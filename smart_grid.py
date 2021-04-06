import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


class SmartGrid(sns.PairGrid):
    def __init__(self, data, *args, log_vars=None, no_diag=False, **kwargs):
        super(SmartGrid, self).__init__(data, *args, **kwargs)
        self.log_vars = log_vars
        if self.log_vars is not None:
            self._log_scale()
        if no_diag:
            self._remove_diag()

    def _map_bivariate(self, func, indices, **kwargs):
        """Draw a bivariate plot on the indicated axes."""
        # This is a hack to handle the fact that new distribution plots don't add
        # their artists onto the axes. This is probably superior in general, but
        # we'll need a better way to handle it in the axisgrid functions.
        from seaborn.distributions import histplot, kdeplot
        if func is histplot or func is kdeplot:
            self._extract_legend_handles = True

        kws = kwargs.copy()  # Use copy as we insert other kwargs
        for i, j in tqdm(indices, desc='Creating Grid', leave=False, unit='subplot'):
            x_var = self.x_vars[j]
            y_var = self.y_vars[i]
            ax = self.axes[i, j]
            if ax is None:  # i.e. we are in corner mode
                continue
            self._plot_bivariate(x_var, y_var, ax, func, **kws)
        self._add_axis_labels()

    def _log_scale(self):
        # TODO: this modifies the diagonal for square grids, but I don't think it should be expected.
        log_vars = [log_var.lower() for log_var in self.log_vars]

        for i, (y_var) in enumerate(self.y_vars):
            # Find y-axes matches to scale
            if y_var.lower() in log_vars:
                # I am picking the axes on the left column
                ax = self.axes[i, 0]
                ax.set_yscale('log')
                if self.data[y_var].max() > 0.1:
                    ax.yaxis.set_major_formatter(plt.LogFormatter(labelOnlyBase=False))
        for j, (x_var) in enumerate(self.x_vars):
            # Find x-axes matches to scale
            if x_var.lower() in log_vars:
                # I am picking the axes on the bottom row
                ax = self.axes[-1, j]
                ax.set_xscale('log')
                if self.data[x_var].max() > 0.1:
                    ax.xaxis.set_major_formatter(plt.LogFormatter(labelOnlyBase=False))

    def _remove_diag(self):
        # TODO: switch to ax removal instead of hiding it
        if self.square_grid:
            for ax in np.diag(self.axes):
                ax.set_visible(False)
            # for indx in zip(*np.diag_indices(len(variables))):
            #     g.axes[indx].remove()
            #     g.axes[indx]=None
        else:
            for i, (y_var) in enumerate(self.y_vars):
                for j, (x_var) in enumerate(self.x_vars):
                    if x_var != y_var:
                        self.axes[i, j].set_visible(False)

    __init__.__doc__ = sns.PairGrid.__init__.__doc__


SmartGrid.__doc__ = sns.PairGrid.__doc__
SmartGrid.__init__.__doc__ = sns.PairGrid.__init__.__doc__

if __name__ == '__main__':
    from regplotter import regplot_log


    def regplot_log_wrap(x, y, log_vars: list = None, err_map: dict = None, data=None,
                         ranges_map=None, delta_map=None, **kwargs):
        logx = x.name in log_vars
        logy = y.name in log_vars
        xerr = err_map.get(x.name)
        yerr = err_map.get(y.name)
        x_range = ranges_map.get(x.name)
        xdelta = delta_map.get(x.name)
        ydelta = delta_map.get(y.name)
        linmix_kws = dict(seed=123456)

        regplot_log(data=data, x=x, y=y, xerr=xerr, yerr=yerr, logx=logx, logy=logy,
                    xdelta=xdelta, ydelta=ydelta, fit_xrange=x_range, linmix_kws=linmix_kws, **kwargs)


    visir = pd.read_csv(r'Data\VISIR_merged_fluxes_TMP.csv', sep=',',
                        skipinitialspace=True, na_values=['#NAME?'])

    x_variables = ['Teff', 'Mstar', 'Lstar', 'logLacc', 'n_13-30']
    y_variables = ['flux_x', 'flux_y', 'flux', 'fwhm_x', 'fwhm_y', 'fwhm']

    err_map = {'flux_x': 'fl_err_x', 'flux_y': 'fl_err_y', 'flux': 'fl_err'}
    ranges_map = {var: (np.nanmin(visir[var]), np.nanmax(visir[var])) for var in x_variables+y_variables}
    plot_log = ['Mstar', 'Lstar', 'Teff', 'flux_x', 'flux_y', 'flux', 'fwhm_x', 'fwhm_y', 'fwhm']

    delta_map = {}
    for col, err_col in err_map.items():
        upp_mask = visir[err_col] > visir[col]
        delta_map[col] = (~upp_mask).astype(int)
        visir.loc[upp_mask, col] = 2 * visir[err_col][upp_mask]

    g = SmartGrid(visir, x_vars=x_variables, y_vars=y_variables, log_vars=plot_log)
    g.map_offdiag(regplot_log_wrap, log_vars=plot_log, err_map=err_map, ranges_map=ranges_map, delta_map=delta_map,
                  data=visir, linmix=True)

    plt.show()

    # THIS SECOND PLOT WILL THROW AN ERROR
    g = SmartGrid(visir, vars=x_variables + y_variables, no_diag=True,
                  diag_sharey=False, corner=True, log_vars=plot_log)
    g.map_lower(regplot_log_wrap, log_vars=plot_log, err_map=err_map, ranges_map=ranges_map, delta_map=delta_map,
                data=visir, linmix=True)

    plt.show()
