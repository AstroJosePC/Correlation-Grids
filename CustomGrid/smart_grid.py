import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


class SmartGrid(sns.PairGrid):
    def __init__(self, data, *args, log_vars=None, no_diag=False, **kwargs):
        super(SmartGrid, self).__init__(data, *args, **kwargs)
        self.log_vars = log_vars
        if self.log_vars is not None:
            self._log_scale()
        if no_diag:
            self._remove_diag()

    def _log_scale(self):
        # TODO: this modifies the diagonal for square grids, but I don't think it should be expected.
        log_vars = [log_var.lower() for log_var in self.log_vars]

        for i, (y_var) in enumerate(self.y_vars):
            # Find y-axes matches to scale
            if y_var.lower() in log_vars:
                print(y_var, i)
                # I am picking the axes on the left column
                ax = self.axes[i, 0]
                ax.set_yscale('log')
        for j, (x_var) in enumerate(self.x_vars):
            # Find x-axes matches to scale
            if x_var.lower() in log_vars:
                print(x_var, j)
                # I am picking the axes on the bottom row
                ax = self.axes[-1, j]
                ax.set_xscale('log')

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
