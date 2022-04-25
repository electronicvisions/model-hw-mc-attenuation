'''
Functions which visualize the distribution of samples in the parameter space.
'''
from typing import Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_1d_empty(ax: plt.Axes, values: np.ndarray,
                  limits: Optional[np.ndarray] = None,
                  kwargs: Optional[Dict[str, Any]] = None) -> Any:
    '''
    Dummy function which creates an empty plot with the desired arguments.

    :param ax: Axes to plot data in.
    :param values: Data to plot the density of.
    :param kwargs: Keyword arguments which should be passed to plotting
        function.
    :return: Result of `plt.Axes.plot()`.
    '''
    del values

    if kwargs is None:
        kwargs = {}

    if limits is not None:
        ax.set_xlim(limits)

    return ax.plot([], [], **kwargs)[0]


def plot_1d_density(ax: plt.Axes, values: np.ndarray,
                    limits: Optional[np.ndarray] = None,
                    kwargs: Optional[Dict[str, Any]] = None) -> Any:
    '''
    Apply Gaussian kernel density estimation to data and plot the result.

    :param ax: Axes to plot data in.
    :param values: Data to plot the density of.
    :param kwargs: Keyword arguments which should be passed to plotting
        function.
    :return: Result of `plt.Axes.plot()`.
    '''
    if kwargs is None:
        kwargs = {}

    density_probability = gaussian_kde(values)

    if limits is None:
        limits = np.array([values.min(), values.max()])

    x_values = np.linspace(limits[0], limits[1], 100)

    return ax.plot(x_values, density_probability(x_values), **kwargs)[0]


def get_xy_1d_hist(values: np.ndarray,
                   limits: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    height, edges = np.histogram(values, range=limits, bins='auto',
                                 density=True)

    x_values = np.repeat(edges, 2)
    y_values = np.concatenate([[0], np.repeat(height, 2), [0]])
    return x_values, y_values


def plot_1d_hist(ax: plt.Axes, values: np.ndarray,
                 limits: Optional[np.ndarray] = None,
                 kwargs: Optional[Dict[str, Any]] = None) -> Any:
    '''
    Plot data in a histogram.

    :param ax: Axes to plot data in.
    :param values: Data to plot the density of.
    :param kwargs: Keyword arguments which should be passed to plotting
        function.
    :return: Result of `plt.Axes.plot()`.
    '''
    if kwargs is None:
        kwargs = {}

    x_values, y_values = get_xy_1d_hist(values, limits)

    return ax.plot(x_values, y_values, **kwargs)[0]


def plot_2d_empty(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray,
                  limits: Optional[np.ndarray] = None,
                  kwargs: Optional[Dict[str, Any]] = None) -> Any:
    del x_values, y_values

    to_be_returned = ax.scatter([], [], **kwargs)

    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    return to_be_returned


def plot_2d_hist(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray,
                 limits: Optional[np.ndarray] = None,
                 kwargs: Optional[Dict[str, Any]] = None) -> Any:
    if kwargs is None:
        kwargs = {}

    if limits is None:
        limits = np.array([[x_values.min(), x_values.max()],
                           [y_values.min(), y_values.max()]])

    x_edges = np.histogram_bin_edges(x_values, bins='auto',
                                     range=limits[0])
    y_edges = np.histogram_bin_edges(y_values, bins='auto',
                                     range=limits[1])

    height, _, _ = np.histogram2d(x_values, y_values,
                                  bins=[x_edges, y_edges], density=True)

    x_values, y_values = np.meshgrid(x_edges, y_edges)
    ax.pcolormesh(x_values, y_values, height.T, edgecolor='face', **kwargs)


def plot_2d_scatter(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray,
                    limits: Optional[np.ndarray] = None,
                    kwargs: Optional[Dict[str, Any]] = None) -> Any:
    if kwargs is None:
        kwargs = {}

    to_be_returned = ax.scatter(x_values, y_values, **kwargs)

    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    return to_be_returned
