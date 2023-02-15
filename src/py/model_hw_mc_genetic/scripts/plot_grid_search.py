#!/usr/bin/env python3
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.ndimage
from model_hw_mc_genetic.attenuation import fit_length_constant
from model_hw_mc_genetic.attenuation.helper import get_bounds


def plot_parameter_space(ax: plt.Axes, data: pd.DataFrame):
    x_name = data.columns[1]
    y_name = data.columns[0]
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


def extract_observable(data: pd.DataFrame, observable: str,
                       target: Optional[pd.DataFrame] = None):
    '''
    Calculate the given observable from the EPSP amplitudes.

    :param data: DataFrame with parameters and EPSP amplitudes.
    :param observable: Observable to extract from the data.
    :param target: DataFrame with target values which should be observed.
        Only needed if observable is 'deviation_amplitudes'.
    :returns: DataFrame with the original parameters and the extracted
        observable.
    '''
    chain_length = data.attrs['length']

    bounds = get_bounds(data)

    # function to extract length constants
    def calculate_length_constants(row: pd.Series) -> float:
        # Extract PSP amplitudes in first compartment
        amplitudes = row['amplitudes'].values.\
            reshape(chain_length, chain_length)[:, 0]
        return fit_length_constant(amplitudes, bounds=bounds)

    # Filter for desired observable
    result = data.loc[:, 'parameters'].copy(deep=True)
    if observable == 'length_constant':
        result['Length Constant'] = \
            data.apply(calculate_length_constants, axis=1)
    elif observable == 'amplitude':
        result['Height'] = data.loc[:, 'amplitudes'].values[:, 0]
    elif observable == 'deviation_amplitudes':
        target = target.mean(axis=0)
        measured = data.loc[:, 'amplitudes']

        # max 30 LSB deviation per compartment
        max_deviation = np.sqrt(30**2 * chain_length)
        deviation = np.linalg.norm(measured - target, axis=1)
        deviation[deviation > max_deviation] = max_deviation
        result['Deviation'] = deviation

    return result


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
                        choices=['length_constant', 'amplitude',
                                 'deviation_amplitudes'],
                        help='Observable to plot. If amplitude is selected, '
                             'the EPSP amplitudes in the first compartment is '
                             'plotted. For the deviation the Euclidean '
                             'distance between the amplitudes is plotted.')
    parser.add_argument('-target',
                        type=str,
                        help='Path to pickled DataFrame with target '
                             'observation. Only needed if `observable` is '
                             '"deviation_amplitudes".')
    args = parser.parse_args()

    input_file = Path(args.grid_search_result)

    fig, axis = plt.subplots()
    target_df = None if args.target is None else pd.read_pickle(args.target)
    filtered_data = extract_observable(pd.read_pickle(input_file),
                                       args.observable,
                                       target=target_df)
    plot_parameter_space(axis, filtered_data)
    fig.savefig(f'{input_file.stem}.png', dpi=300)
