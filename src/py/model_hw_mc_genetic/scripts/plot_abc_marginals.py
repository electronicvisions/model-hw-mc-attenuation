#!/usr/bin/env python3
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model_hw_mc_genetic.plotting.marginals import plot_marginals


def main(posterior_dfs: List[pd.DataFrame],
         labels: Optional[List[str]] = None) -> plt.Figure:
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
    n_params = posterior_dfs[0]['parameters'].shape[1]

    n_columns = (n_params + 1) // 2
    fig, axs = plt.subplots(2, n_columns,
                            figsize=np.array([n_columns, 2]) * 4,
                            sharey='row', tight_layout=True)

    plot_marginals(axs.flatten()[:-1], posterior_dfs, labels)
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
    args = parser.parse_args()

    # read data
    figure = main(
        [pd.read_pickle(pos_file) for pos_file in args.posterior_files],
        labels=args.labels)
    figure.savefig('abc_marginals.svg')
