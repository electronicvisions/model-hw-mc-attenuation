#!/usr/bin/env python3
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model_hw_mc_genetic.plotting.density import plot_1d_hist
from model_hw_mc_genetic.helper import AttributeNotIdentical, \
    get_identical_attr, get_parameter_limits


def plot_parameter(axis,
                   parameter_name: str,
                   posterior_dfs: List[pd.DataFrame],
                   labels: Optional[List[str]] = None,
                   plot_combined: bool = False) -> None:
    '''
    Plot the one-dimensional marginals of the given parameter.

    :param axis: Axis in which to plot the marginals are plotted.
    :param parameter_name: name of the parameter to plot.
    :param posterior_dfs: Data Frames with posterior samples for which to plot
        the one-dimensional marginals.
    :param labels: Labels for the different DataFrames.
    :param plot_combined: Plot the one-dimensional marginals for all samples
        from the different posteriors combined.
    '''
    labels = np.arange(len(posterior_dfs)) if labels is None else labels

    limits = get_parameter_limits([df['parameters'] for df in posterior_dfs],
                                  parameter_name)

    # plot different distributions
    for n_pos, (samples, label) in enumerate(zip(posterior_dfs, labels)):

        # Only use provided labels if the combined result is not plotted
        if plot_combined:
            kwargs = {'color': 'k', 'alpha': 0.2}
            kwargs.update({'label': 'single'} if n_pos == 0 else {})
        else:
            kwargs = {'label': label}

        plot_1d_hist(axis,
                     samples[('parameters', parameter_name)],
                     limits=limits, **kwargs)

    if plot_combined:
        all_samples = pd.concat(posterior_dfs)
        samples = all_samples.loc[:, ('parameters', parameter_name)]
        plot_1d_hist(axis, samples, limits=limits, label='combined')


def plot_original_parameters(axes: plt.Axes,
                             posterior_dfs: List[pd.DataFrame]) -> None:
    '''
    Add a vertical line with the original parameter in each axis.

    The line is only added if all posteriors are based on the identical
    original parameters.

    :param axes: Axes to which the vertical lines. The axes are assumed to
        be in the same order as the parameters.
    :param posterior_dfs: DataFrames with samples from the posteriors. From
        their attributes the initial target is extracted and from the target
        the parameter.
    '''
    try:
        target_dfs = [pd.read_pickle(df.attrs['target_file']) for df in
                      posterior_dfs]
    except KeyError:
        return
    try:
        original_parameters = get_identical_attr(target_dfs, 'parameters')
    except AttributeNotIdentical:
        return

    if posterior_dfs[0]['parameters'].shape[0] == 2:
        # Parameters set globally
        original_parameters = original_parameters[[0, -1]]

    for ax, parameter_value in zip(axes.flatten(), original_parameters):
        ax.axvline(parameter_value, c='k', alpha=0.5, ls='-')


def plot_marginals(posterior_dfs: List[pd.DataFrame],
                   labels: Optional[List[str]] = None,
                   plot_combined_posterior: bool = False) -> plt.Figure:
    '''
    Plot the 1d-marginals of the provided posterior samples.

    The one-dimensional distribution of each individual parameter is plotted
    in a separate plot in the figure.
    If the original parameters to which the posteriors are conditioned can be
    extracted, they are marked in the individual plots.

    :param posterior_dfs: Data Frames with posterior samples for which to plot
        the one-dimensional marginals.
    :param labels: Labels for the different DataFrames.
    :returns: Figure with the one-dimensional marginals.
    '''
    parameter_names = posterior_dfs[0]['parameters'].columns.to_list()

    # Create figure
    n_columns = (len(parameter_names) + 1) // 2
    fig, axs = plt.subplots(2, n_columns,
                            figsize=np.array([n_columns, 2]) * 4,
                            sharey='row', tight_layout=True)

    plot_original_parameters(axs, posterior_dfs)

    # plot parameter distribution

    for ax, parameter in zip(axs.flatten(), parameter_names):
        plot_parameter(ax, parameter, posterior_dfs,
                       labels=labels,
                       plot_combined=plot_combined_posterior)
        ax.set_xlabel(parameter)

    axs.flatten()[(axs.size - 1) // 2].legend()

    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot the 1d-marginals of the provided posterior samples. '
                    'The one-dimensional distribution of each individual '
                    'parameter is plotted in a separate plot in the figure.')
    parser.add_argument('posterior_files',
                        type=str,
                        nargs='+',
                        help='Path to pickled DataFrames with samples drawn '
                             'from the posterior.')
    parser.add_argument('-labels',
                        nargs='+',
                        type=str,
                        help='Label for each posterior_files.')
    parser.add_argument('--plot_combined_posterior',
                        help='Also plot all posteriors combined in a single '
                             'histogram.',
                        action='store_true')
    args = parser.parse_args()

    # read data
    figure = plot_marginals(
        [pd.read_pickle(pos_file) for pos_file in args.posterior_files],
        labels=args.labels,
        plot_combined_posterior=args.plot_combined_posterior)
    figure.savefig('abc_marginals.svg')
