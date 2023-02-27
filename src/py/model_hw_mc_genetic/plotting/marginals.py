#!/usr/bin/env python3
from typing import List, Optional, Callable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model_hw_mc_genetic.helper import get_parameter_limits
from model_hw_mc_genetic.plotting.density import plot_1d_hist


def plot_parameter(axis,
                   parameter_name: str,
                   posterior_dfs: List[pd.DataFrame],
                   labels: Optional[List[str]] = None,
                   plotting_func: Callable = plot_1d_hist,
                   **kwargs) -> None:
    '''
    Plot the one-dimensional marginals of the given parameter.

    :param axis: Axis in which to plot the marginals are plotted.
    :param parameter_name: Name of the parameter to plot.
    :param posterior_dfs: Data Frames with posterior samples for which to plot
        the one-dimensional marginals.
    :param labels: Labels for the different DataFrames.
    :param plotting_func: Function used to plot the marginal distribution.
    '''
    labels = np.arange(len(posterior_dfs)) if labels is None else labels

    limits = get_parameter_limits([df['parameters'] for df in posterior_dfs],
                                  parameter_name)

    for samples, label in zip(posterior_dfs, labels):
        default_kwargs = {'label': label}
        default_kwargs.update(kwargs)

        plotting_func(axis, samples[('parameters', parameter_name)],
                      limits=limits, **default_kwargs)


def plot_marginals(axs: List[plt.Axes],
                   posterior_dfs: List[pd.DataFrame],
                   original_parameters: Optional[np.ndarray] = None,
                   labels: Optional[List[str]] = None):

    '''
    Plot the 1d-marginals of the provided posterior samples.

    The one-dimensional distribution of each individual parameter is plotted
    in a separate plot in the figure.
    If the original parameters to which the posteriors are conditioned can be
    extracted, they are marked in the individual plots.

    :param axs: Axes in wich to plot the marginals.
    :param posterior_dfs: Data Frames with posterior samples for which to plot
        the one-dimensional marginals.
    :param original_parameters: Parameters used to record the observation on
        which the posteriors where conditioned.
    :param labels: Labels for the different DataFrames.
    '''
    if original_parameters is not None:
        for ax, parameter_value in zip(axs.flatten(), original_parameters):
            ax.axvline(parameter_value, c='k', alpha=0.5, ls='-')

    parameter_names = posterior_dfs[0]['parameters'].columns.to_list()
    for ax, parameter in zip(axs.flatten(), parameter_names):
        plot_parameter(ax, parameter, posterior_dfs, labels=labels)
        ax.set_xlabel(parameter)
