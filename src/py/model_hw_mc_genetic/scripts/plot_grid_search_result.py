#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.ndimage


def plot_parameter_space(ax: plt.Axes, data: pd.DataFrame):
    x_name = data.columns[0]
    y_name = data.columns[1]
    z_name = data.columns[2]

    # arrange data
    data = data.pivot(index=y_name, columns=x_name,
                      values=z_name).sort_index(ascending=False)

    x_data, y_data = np.meshgrid(data.columns, data.index)

    # colormap
    cmap = plt.get_cmap('viridis')
    im_plot = ax.pcolormesh(x_data, y_data, data.values, cmap=cmap,
                            edgecolor='face', shading='nearest')
    color_bar = ax.figure.colorbar(im_plot, ax=ax)
    color_bar.set_label(z_name)

    # contour plot
    values = scipy.ndimage.filters.gaussian_filter(data.values, 3)

    contour = ax.contour(x_data, y_data, values, colors='w', linewidths=1)
    ax.clabel(contour, inline=True)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    return im_plot


if __name__ == '__main__':
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot a heat map of the selected observable in dependency '
                    'on the leak and inter-compartment conductance. On top '
                    'smoothed contour lines where the observable is constant '
                    'are plotted.')
    parser.add_argument('grid_search_result',
                        type=str,
                        help='Path to pickled DataFrame with parameters and '
                             'measurement results.')
    parser.add_argument('-observable',
                        type=str,
                        default='length_constant',
                        choices=['length_constant', 'height'],
                        help='Observable to plot. If height is selected, the '
                             'EPSP height in the first compartment is '
                             'plotted.')
    args = parser.parse_args()

    input_file = Path(args.grid_search_result)

    fig, axis = plt.subplots()
    grid_data = pd.read_pickle(input_file)

    # Filter for desired observable
    filtered_data = grid_data.iloc[:, [0, 1]].copy(deep=True)
    if args.observable == 'length_constant':
        filtered_data['Length Constant'] = grid_data.loc[:, 'length_constant']
    elif args.observable == 'height':
        filtered_data['Height'] = \
            np.stack(grid_data.loc[:, 'heights'])[:, 0]

    plot_parameter_space(axis, filtered_data)

    fig.savefig(f'{input_file.stem}.png', dpi=300)
