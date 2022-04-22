#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.ndimage


def plot_parameter_space(ax: plt.Axes, data: pd.DataFrame):
    x_name = data.columns[0]
    y_name = data.columns[1]

    # arrange data
    data = data.pivot(index=y_name, columns=x_name,
                      values='length_constant').sort_index(ascending=False)

    x_data, y_data = np.meshgrid(data.columns, data.index)

    # colormap
    cmap = plt.get_cmap('viridis')
    im_plot = ax.pcolormesh(x_data, y_data, data.values, cmap=cmap,
                            edgecolor='face', shading='nearest')
    color_bar = ax.figure.colorbar(im_plot, ax=ax)
    color_bar.set_label('length constant')

    # contour plot
    values = scipy.ndimage.filters.gaussian_filter(data.values, 3)

    contour = ax.contour(x_data, y_data, values, colors='w', linewidths=1)
    ax.clabel(contour, inline=True)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    return im_plot


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot a heat map of the length constant in dependency on '
                    'the leak and inter-compartment conductance. On top '
                    'smoothed contour lines of constant length constant are '
                    'plotted.')
    parser.add_argument('grid_search_result',
                        type=str,
                        help='Path to pickled DataFrame with parameters and '
                             'measurement results.')
    args = parser.parse_args()

    fig, axis = plt.subplots()
    grid_data = pd.read_pickle(args.grid_search_result)

    plot_parameter_space(axis, grid_data)

    fig.savefig('grid_search.png', dpi=300)
