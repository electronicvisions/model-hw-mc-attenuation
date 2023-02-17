#!/usr/bin/env python3
from typing import List
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model_hw_mc_genetic.plotting.pairplot import pairplot, create_axes_grid
from model_hw_mc_genetic.plotting.density import plot_2d_scatter
from model_hw_mc_genetic.helper import AttributeNotIdentical, \
    get_identical_attr


scatter_2d = partial(plot_2d_scatter, max_points=800, alpha=0.3,
                     edgecolor='none', s=12)


def plot_pairplot(samples_dfs: List[pd.DataFrame]) -> plt.Figure:
    '''
    Create pairplots for the given samples.

    Plot the pairwise distribution of the samples in the same axes.

    :param samples_dfs: List of DataFrames with samples for which to plot the
        distributions.
    :returns: Figure.
    '''
    ###########################################################################
    # Load data
    ###########################################################################
    param_names = list(samples_dfs[0]['parameters'].columns)
    n_param = len(param_names)

    try:
        target_dfs = [pd.read_pickle(df.attrs['target_file']) for df in
                      samples_dfs]
        original_parameters = get_identical_attr(target_dfs, 'parameters')
    except (KeyError, AttributeNotIdentical):
        original_parameters = None

    # restrict original parameters to two values if the given samples represent
    # global parameters
    if original_parameters is not None and \
            len(samples_dfs[0]['parameters'].columns.to_list()) == 2:
        chain_length = (len(original_parameters) + 1) // 2
        g_leak = np.mean(original_parameters[:chain_length])
        g_icc = np.mean(original_parameters[chain_length:])
        original_parameters = [g_leak, g_icc]

    ###########################################################################
    # Plotting
    ###########################################################################
    size = 2 * np.ones(2) * n_param if n_param > 2 else [8, 8]
    fig = plt.figure(figsize=size, tight_layout=True)
    base_gs = fig.add_gridspec(1)
    axes = create_axes_grid(base_gs[0], n_param)
    for samples_df in samples_dfs:
        pairplot(axes, samples_df['parameters'].values, labels=param_names,
                 limits=samples_df.attrs['limits'],
                 plot_2d_dist=scatter_2d,
                 target_params=original_parameters)
    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='For each DataFrame with posterior samples display the '
                    'pairwise distribution of the samples.')
    parser.add_argument('pos_samples_files',
                        nargs='+',
                        type=str,
                        help='Path to pickled DataFrames with samples drawn '
                             'from the posteriors.')
    args = parser.parse_args()

    pos_samples_dfs = []
    for filename in args.pos_samples_files:
        pos_samples_dfs.append(pd.read_pickle(filename))

    figure = plot_pairplot(pos_samples_dfs)

    figure.savefig('pairplot.png')
