from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.ndimage
from model_hw_mc_attenuation import Observation
from model_hw_mc_attenuation.extract import extract_observation


def plot_heat_map(ax: plt.Axes, data: pd.DataFrame, **kwargs
                  ) -> matplotlib.collections.QuadMesh:
    '''
    Plot a heat map in the given axis.

    Keyword arguments are passed to pcolormesh

    :param ax: Axis in which to plot the heat map.
    :param data: Data for which to plot the heat map. The x values are assumed
        to be in the first column, the y values in the second column and the z
        values in the third column.
    :returns: Artist created by :meth:`plt.Axes.pcolormesh`.
    '''
    x_name = data.columns[0]
    y_name = data.columns[1]
    z_name = data.columns[2]

    # arrange data
    data = data.pivot(index=y_name, columns=x_name,
                      values=z_name).sort_index(ascending=False)

    x_data, y_data = np.meshgrid(data.columns, data.index)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    args = {'cmap': 'cividis', 'edgecolor': 'face', 'shading': 'nearest'}
    args.update(kwargs)

    return ax.pcolormesh(x_data, y_data, data.values, **args)


def plot_contour_lines(ax: plt.Axes, data: pd.DataFrame, *,
                       smooth_sigma: float = 0, **kwargs
                       ) -> matplotlib.contour.QuadContourSet:
    '''
    Add contour plot to the given axis.

    Keyword arguments are passed to :meth:`plt.Axes.contour`.

    :param ax: Axis in which to plot the contour lines.
    :param data: Data for which to plot the heat map. The x values are assumed
        to be in the first column, the y values in the second column and the z
        values in the third column.
    :param smooth_sigma: Standard deviation for Gaussian kernel used for
        smoothing.
    :returns: Created contour lines.
    '''
    x_name = data.columns[0]
    y_name = data.columns[1]
    z_name = data.columns[2]

    # arrange data
    data = data.pivot(index=y_name, columns=x_name,
                      values=z_name).sort_index(ascending=False)

    x_data, y_data = np.meshgrid(data.columns, data.index)
    values = data.values
    if smooth_sigma > 0:
        values = scipy.ndimage.filters.gaussian_filter(values, smooth_sigma)

    default_kwargs = {'colors': 'w', 'linewidths': 1}
    default_kwargs.update(kwargs)
    return ax.contour(x_data, y_data, values, **default_kwargs)


def create_obs_dataframe(data: pd.DataFrame, observation: Observation,
                         target: Optional[np.ndarray] = None) -> pd.DataFrame:
    '''
    Create DataFrame with the parameters and the given observation.

    :param data: DataFrame with parameters and EPSP amplitudes.
    :param observation: Observable to extract from the data.
    :param target: One-dimensional array with target amplitudes.
        Only needed if observable is 'deviation_amplitudes'.
    :returns: DataFrame with the original parameters and the extracted
        observable.
    '''
    result = data.loc[:, 'parameters'].copy(deep=True)
    obs = extract_observation(data['amplitudes'], observation, target)

    if observation == Observation.AMPLITUDES_DISTANCE:
        # max 30 LSB deviation per compartment
        max_deviation = np.sqrt(30**2 * data.attrs['length'])
        obs[obs > max_deviation] = max_deviation
    result[observation.name.lower()] = obs

    return result
