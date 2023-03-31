#!/usr/bin/env python3
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_hw_mc_attenuation import Observation
from model_hw_mc_attenuation.plotting.grid_search import \
    create_obs_dataframe, plot_heat_map, plot_contour_lines


def main(ax: plt.Axes, data: pd.DataFrame, observation: Observation,
         target: Optional[np.ndarray] = None) -> None:
    '''
    Create heat map of the given observation.

    :param ax: Axis in which to plot the heat map.
    :param data: DataFrame with parameters and EPSP amplitudes.
    :param observation: Observable to plot.
    :param target: One-dimensional array with target amplitudes.
        Only needed if the observation is 'deviation_amplitudes'.
    '''

    filtered_data = create_obs_dataframe(
        data, observation, target=None if target is None else target.mean(0))

    im_plot = plot_heat_map(ax, filtered_data)

    color_bar = ax.figure.colorbar(im_plot, ax=ax)
    color_bar.set_label(observation.name.lower())

    contour = plot_contour_lines(ax, filtered_data, smooth_sigma=3)
    ax.clabel(contour, inline=True)


if __name__ == '__main__':
    from pathlib import Path
    import argparse

    # only one-dimensional observations can be plotted.
    possible_obs = [Observation.LENGTH_CONSTANT, Observation.AMPLITUDE_00,
                    Observation.AMPLITUDES_DISTANCE]

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
                        default=Observation.LENGTH_CONSTANT.name.lower(),
                        choices=[obs.name.lower() for obs in possible_obs],
                        help='Observable to plot.')
    parser.add_argument('-target',
                        type=str,
                        help='Path to pickled DataFrame with target '
                             'observation. Only needed if `observable` is '
                             '"amplitudes_distance".')
    args = parser.parse_args()

    input_file = Path(args.grid_search_result)

    fig, axis = plt.subplots()
    target_df = None if args.target is None else pd.read_pickle(args.target)
    main(axis, pd.read_pickle(input_file),
         Observation[args.observable.upper()],
         target=target_df)
    fig.savefig(f'{input_file.stem}.png')
