from inspect import signature

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatterSciNotation
from tqdm import tqdm

# mpl.rcParams['axes.formatter.min_exponent'] = 5

"""
TODO: Make a better log formatter that
- uses base^exp format when axis span >3 decades
- picks lower number of labels between bases 
- prefers to label numbers |>1| as whole numbers (fractions are less attractive)
- is consistent on what format to use on a single axes; base^exp VS integer 
"""


class NewFormatter(plt.LogFormatter):
    def _num_to_string(self, x, vmin, vmax):
        if x > 10000:
            s = '%1.0e' % x
        else:
            s = self._pprint_val(x, vmax - vmin)
        return s


class SmartGrid(sns.PairGrid):
    def __init__(self, data, *args, log_vars=None, no_diag=False, labels_map: dict = None, pass_subsets: dict = None,
                 **kwargs):
        self.labels_map = labels_map
        markers = kwargs.pop('markers', None)
        linestyles = kwargs.pop('linestyles', None)
        super(SmartGrid, self).__init__(data, *args, **kwargs)

        self.pass_subsets = dict() if pass_subsets is None else pass_subsets

        if markers is not None:
            if 'marker' not in self.pass_subsets:
                self.pass_subsets['marker'] = markers

        if linestyles is not None:
            if 'linestyle' not in self.pass_subsets:
                self.pass_subsets['linestyle'] = linestyles

        for key, value in self.pass_subsets.items():
            if self.hue_names is None:
                n_items = 1
            else:
                n_items = len(self.hue_names)
            to_pass = value
            if not isinstance(value, list):
                to_pass = [value] * n_items
            if len(to_pass) != n_items:
                raise ValueError('Each item to pass must be a singleton or a list of '
                                 f'objects for each level of the hue variable: {key}')
            self.hue_kws = {key: to_pass, **self.hue_kws}

        self.log_vars = log_vars
        if self.log_vars is not None:
            self._log_scale()
        if no_diag:
            self._remove_diag()

    def add_legend(self, legend_data=None, title=None, label_order=None,
                   adjust_subtitles=False, **kwargs):
        """Draw a legend, maybe placing it outside axes and resizing the figure.

        Parameters
        ----------
        legend_data : dict
            Dictionary mapping label names (or two-element tuples where the
            second element is a label name) to matplotlib artist handles. The
            default reads from ``self._legend_data``.
        title : string
            Title for the legend. The default reads from ``self._hue_var``.
        label_order : list of labels
            The order that the legend entries should appear in. The default
            reads from ``self.hue_names``.
        adjust_subtitles : bool
            If True, modify entries with invisible artists to left-align
            the labels and set the font size to that of a title.
        kwargs : key, value pairings
            Other keyword arguments are passed to the underlying legend methods
            on the Figure or Axes object.

        Returns
        -------
        self : Grid instance
            Returns self for easy chaining.

        """
        if label_order is None and self.hue_names == ['_nolegend_']:
            self.hue_names = None
            super().add_legend(legend_data=None, title=None, label_order=None, adjust_subtitles=False, **kwargs)
            self.hue_names = ['_nolegend_']
        else:
            super().add_legend(legend_data=None, title=None, label_order=None, adjust_subtitles=False, **kwargs)
        return self

    def _map_bivariate(self, func, indices, **kwargs):
        """Draw a bivariate plot on the indicated axes."""
        # This is a hack to handle the fact that new distribution plots don't add
        # their artists onto the axes. This is probably superior in general, but
        # we'll need a better way to handle it in the axisgrid functions.
        from seaborn.distributions import histplot, kdeplot
        if func is histplot or func is kdeplot:
            self._extract_legend_handles = True

        kws = kwargs.copy()  # Use copy as we insert other kwargs
        for i, j in tqdm(indices, desc='Creating Grid', leave=True, unit=' subplot', disable=False):
            x_var = self.x_vars[j]
            y_var = self.y_vars[i]
            ax = self.axes[i, j]
            if ax is None:  # i.e. we are in corner mode
                continue
            self._plot_bivariate(x_var, y_var, ax, func, **kws)
        self._add_axis_labels()

    def _plot_bivariate_iter_hue(self, x_var, y_var, ax, func, **kwargs):
        """Draw a bivariate plot while iterating over hue subsets."""
        kwargs = kwargs.copy()
        if str(func.__module__).startswith("seaborn"):
            kwargs["ax"] = ax
        else:
            plt.sca(ax)

        if x_var == y_var:
            axes_vars = [x_var]
        else:
            axes_vars = [x_var, y_var]

        hue_grouped = self.data.groupby(self.hue_vals)
        for k, label_k in enumerate(self._hue_order):
            # print('RESPONSIVE!', label_k, x_var, y_var)
            kws = kwargs.copy()

            # Attempt to get data for this level, allowing for empty
            try:
                data_k = hue_grouped.get_group(label_k)
            except KeyError:
                data_k = pd.DataFrame(columns=axes_vars,
                                      dtype=float)

            if self._dropna:
                data_k = data_k[axes_vars].dropna()

            x = data_k[x_var]
            y = data_k[y_var]

            for kw, val_list in self.hue_kws.items():
                kws[kw] = val_list[k]
            kws.setdefault("color", self.palette[k])
            if self._hue_var is not None:
                kws["label"] = label_k

            if "data" in signature(func).parameters:
                kws['data'] = data_k

            if str(func.__module__).startswith("seaborn"):
                func(x=x, y=y, **kws)
            else:
                func(x, y, **kws)

        self._update_legend_data(ax)

    def _log_scale(self, formatter=None):
        """
        Notes on log formatters:
        Increase  minor_thresholds[0] to trigger minor tick labeling, decrease to toggle it off

        :param formatter:
        :return:
        """
        # TODO: this modifies the diagonal for square grids, but I don't think it should be expected.
        log_vars = [log_var.lower() for log_var in self.log_vars]
        # if formatter is None:
        #     formatter = LogFormatterSciNotation()
        for i, (y_var) in enumerate(self.y_vars):
            # Find y-axes matches to scale
            if y_var.lower() in log_vars:
                # I am picking the axes on the left column
                ax = self.axes[i, 0]
                ax.set_yscale('log')
                # if self.data[y_var].max() > 0.1:
                # formatter_y = NewFormatter(labelOnlyBase=False, minor_thresholds=(1.2, 0.5))
                formatter_y = LogFormatterSciNotation(minor_thresholds=(1.1, 0.4))
                ax.yaxis.set_major_formatter(formatter_y)
                ax.yaxis.set_minor_formatter(formatter_y)
        for j, (x_var) in enumerate(self.x_vars):
            # Find x-axes matches to scale
            if x_var.lower() in log_vars:
                # I am picking the axes on the bottom row
                ax = self.axes[-1, j]
                ax.set_xscale('log')
                # if self.data[x_var].max() > 0.1:
                formatter_x = LogFormatterSciNotation(minor_thresholds=(1.1, 0.4))
                # formatter_x = NewFormatter(labelOnlyBase=False, minor_thresholds=(0.8, 0.1))
                ax.xaxis.set_major_formatter(formatter_x)
                ax.xaxis.set_minor_formatter(formatter_x)

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

    def _add_axis_labels(self):
        """Add labels to the left and bottom Axes."""
        for ax, label in zip(self.axes[-1, :], self.x_vars):
            if self.labels_map:
                ax.set_xlabel(self.labels_map.get(label, label))
            else:
                ax.set_xlabel(label)
        for ax, label in zip(self.axes[:, 0], self.y_vars):
            if self.labels_map:
                ax.set_ylabel(self.labels_map.get(label, label))
            else:
                ax.set_ylabel(label)
        if self._corner:
            self.axes[0, 0].set_ylabel("")

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
    ranges_map = {var: (np.nanmin(visir[var]), np.nanmax(visir[var])) for var in x_variables + y_variables}
    plot_log = ['Mstar', 'Lstar', 'Teff', 'flux_x', 'flux_y', 'flux', 'fwhm_x', 'fwhm_y', 'fwhm']

    delta_map = {}
    for col, err_col in err_map.items():
        upp_mask = visir[err_col] > visir[col]
        delta_map[col] = (~upp_mask).astype(int)
        # REVISIT: determine whether x2 sigma or not is better
        visir.loc[upp_mask, col] = visir[err_col][upp_mask]

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
