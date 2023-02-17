'''
Functions which visualize the distribution of samples in the parameter space.
'''
from typing import Optional, Any, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import scipy.ndimage


def plot_1d_empty(ax: plt.Axes, values: np.ndarray,
                  limits: Optional[np.ndarray] = None, **kwargs) -> plt.Line2D:
    '''
    Dummy function which creates an empty plot with the desired arguments.

    :param ax: Axes to plot data in.
    :param values: Data to plot the density of.
    :param limits: Limits to use for the input data.
    :return: Line artist.
    '''
    del values

    if limits is not None:
        ax.set_xlim(limits)

    return ax.plot([], [], **kwargs)[0]


def plot_1d_density(ax: plt.Axes, values: np.ndarray,
                    limits: Optional[np.ndarray] = None, **kwargs
                    ) -> plt.Line2D:
    '''
    Apply Gaussian kernel density estimation to data and plot the result.

    :param ax: Axes to plot data in.
    :param values: Data to plot the density of.
    :param limits: Limits to use when plotting the estimated density.
    :return: Line artist.
    '''

    density_probability = gaussian_kde(values)

    if limits is None:
        limits = np.array([values.min(), values.max()])

    x_values = np.linspace(limits[0], limits[1], 100)

    return ax.plot(x_values, density_probability(x_values), **kwargs)[0]


def get_xy_1d_hist(values: np.ndarray,
                   limits: Optional[np.ndarray] = None,
                   bins: Union[int, np.ndarray, str] = 100,
                   density: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Create data for a line plot which represents a histogram.

    :param values: Data for which to create the x-/y-values of a histogram.
    :param limits: Limits which are used when determining the height of the
        different bins.
    :param bins: Bins argument passed to `np.histogram`.
    :param density: Calculate density of samples in each bin if `True`, else
        calculate the number of samples in each bin.
    :return: Tuple of x_values and y_values which represent the histogram (as
        a line plot).
    '''
    height, edges = np.histogram(values, range=limits, bins=bins,
                                 density=density)

    x_values = np.repeat(edges, 2)
    y_values = np.concatenate([[0], np.repeat(height, 2), [0]])
    return x_values, y_values


def plot_1d_hist(ax: plt.Axes, values: np.ndarray,
                 limits: Optional[np.ndarray] = None,
                 bins: Union[int, np.ndarray, str] = 100,
                 density: bool = False, **kwargs) -> plt.Line2D:
    '''
    Plot data in a histogram.

    :param ax: Axes to plot data in.
    :param values: Data to plot the density of.
    :param limits: Limits to use for the input data.
    :param bins: Bins argument passed to `np.histogram`.
    :param density: Plot density of samples in each bin if `True`, else plot
        number of samples in each bin.
    :return: Line artist.
    '''
    x_values, y_values = get_xy_1d_hist(values, limits, bins=bins,
                                        density=density)

    return ax.plot(x_values, y_values, **kwargs)[0]


def plot_1d_gauss(ax: plt.Axes, values: np.ndarray,
                  limits: Optional[np.ndarray] = None,
                  bins: Union[int, np.ndarray, str] = 'auto', **kwargs
                  ) -> plt.Line2D:
    '''
    Plot data in a histogram.

    :param ax: Axes to plot data in.
    :param values: Data to plot the density of.
    :param limits: Limits which are used when determining the height of the
        different bins.
    :param bins: Bins argument passend to `np.histogram`.
    :param density: Plot number of samples in each bin if `False`, else plot
        density of samples in each bin.
    :return: Line artist.
    '''
    hist, bin_edges = np.histogram(values, density=True, bins=bins,
                                   range=limits)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    def gauss(x_value, scale, mean, sigma):
        return scale * np.exp(-(x_value - mean)**2 / (2 * sigma**2))

    start_values = [1., np.mean(values), np.std(values)]

    coeff = curve_fit(gauss, bin_centres, hist, p0=start_values)[0]

    # Get the fitted curve
    hist_fit = gauss(bin_centres, *coeff)

    return ax.plot(bin_centres, hist_fit, **kwargs)


def plot_2d_empty(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray,
                  limits: Optional[np.ndarray] = None, **kwargs) -> Any:
    '''
    Dummy function which creates an empty plot with the desired arguments.

    :param ax: Axes to plot data in.
    :param x_value: Data on the x-axis.
    :param y_value: Data on the y-axis.
    :param limits: Limits of the input data. Shaped (2, 2) with the limits
        for the x-values in the first row and the values for the y-data
        in the second.
    :return: PathCollection returned by `ax.scatter`
    '''
    del x_values, y_values

    to_be_returned = ax.scatter([], [], **kwargs)

    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    return to_be_returned


def plot_2d_hist(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray,
                 limits: Optional[np.ndarray] = None, **kwargs) -> Any:
    '''
    Create a two-dimensional histogram of the input data.

    :param ax: Axes to plot data in.
    :param x_value: Data on the x-axis.
    :param y_value: Data on the y-axis.
    :param limits: Limits of the input data. Shaped (2, 2) with the limits
        for the x-values in the first row and the values for the y-data
        in the second.
    :return: Artist returned by `ax.pcolormesh`.
    '''
    density, x_edges, y_edges = \
        np.histogram2d(x_values, y_values, range=limits, bins=100)
    x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)

    ax.pcolormesh(x_mesh, y_mesh, density.T, edgecolor='face', **kwargs)


def plot_2d_scatter(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray,
                    limits: Optional[np.ndarray] = None,
                    max_points: Optional[int] = None, **kwargs) -> Any:
    '''
    Plot the given values on the given axes.

    :param ax: Axes to plot data in.
    :param x_value: Data on the x-axis.
    :param y_value: Data on the y-axis.
    :param limits: Limits of the input data. Shaped (2, 2) with the limits
        for the x-values in the first row and the values for the y-data
        in the second.
    :param max_points: Maximum number of points to plot.
    :return: PathCollection returned by `ax.scatter`
    '''

    if max_points is not None and max_points < len(x_values):
        indices = np.random.choice(np.arange(len(x_values)), size=max_points,
                                   replace=False)
        x_values = x_values[indices]
        y_values = y_values[indices]

    to_be_returned = ax.scatter(x_values, y_values, **kwargs)

    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    return to_be_returned


def plot_2d_enumerate(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray,
                      limits: Optional[np.ndarray] = None, **kwargs) -> Any:
    '''
    Add annotations with the samples enumerated in increasing order.

    :param ax: Axes to plot data in.
    :param x_value: Data on the x-axis.
    :param y_value: Data on the y-axis.
    :param limits: Limits of the input data. Shaped (2, 2) with the limits
        for the x-values in the first row and the values for the y-data
        in the second.
    :return: Numpy array of dtype object with one element which is a list of
        the created annotations.
    '''

    my_style = {'bbox': {'boxstyle': 'circle, pad=0.3', 'fc': 'w', 'ec': 'k',
                         'lw': 2},
                'size': 'small'}
    my_style.update(kwargs)

    annotations = []
    for n_sample, xy_coord in enumerate(zip(x_values, y_values)):
        annotations.append(ax.annotate(str(n_sample), xy=xy_coord, ha="center",
                                       va="center", **my_style))

    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    to_be_returned = np.empty(1, dtype=object)
    to_be_returned[0] = annotations
    return to_be_returned


def _fit_kde(x_values: np.ndarray, y_values: np.ndarray,
             limits: Optional[np.ndarray] = None,
             bins: Union[int, np.ndarray, Tuple] = 100) -> Tuple:
    '''
    Fit a kernel density estimate to the given data.

    The contour intervals for the 0.68 and 0.95 confidence interval a plotted.

    :param ax: Axes to plot data in.
    :param x_value: Data on the x-axis.
    :param y_value: Data on the y-axis.
    :param limits: Limits of the input data. Shaped (2, 2) with the limits
        for the x-values in the first row and the values for the y-data
        in the second.
    :param bins: Bin argument passed to `np.histogram2d`.
    :return: Artist returned by `ax.contour`
    '''

    _, x_edges, y_edges = np.histogram2d(x_values, y_values, range=limits,
                                         bins=bins)
    x_mesh, y_mesh = np.meshgrid((x_edges[1:] + x_edges[:-1]) / 2,
                                 (y_edges[1:] + y_edges[:-1]) / 2)

    kde = gaussian_kde([x_values, y_values])
    values = kde(np.vstack([x_mesh.ravel(), y_mesh.ravel()])).T.\
        reshape(x_mesh.shape).T
    values = scipy.ndimage.filters.gaussian_filter(values, 3)

    return x_mesh, y_mesh, values


def _find_levels(values: np.array, levels: np.array) -> np.array:
    """
    Assign values to one of the provided levels.

    The levels are assumed to represent a cumulative probability. The values
    in `values` are assigned to the levels of the cumulative probabilities.

    :param values: Values which should be assigned to the levels. The values
        are assumed to represent some kind of probability/count.
    :param levels: Levels to which to assign the values. Need to be in the
        interval (0, 1).
    :return: Level of each value in `values`. The output has the same shape as
        `values`.
    """
    # inspired by sbi package
    levels = np.asarray(levels)
    if not np.all(levels < 1) and np.all(levels > 0):
        raise ValueError('`levels` need to be in range (0, 1).')

    # sort values
    idx_sort = values.argsort(axis=None)[::-1]  # save in order to undo later
    sorted_values = values.flatten()[idx_sort]

    # cumulative probabilities
    cum_probs = sorted_values.cumsum()
    cum_probs /= cum_probs[-1]  # normalize

    # create contours at levels
    assigned_levels = np.ones_like(cum_probs)
    for level in np.sort(levels)[::-1]:
        assigned_levels[cum_probs <= level] = level

    # undo sorting and convert to original shape
    return assigned_levels[idx_sort.argsort()].reshape(values.shape)


def plot_2d_contours(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray,
                     limits: Optional[np.ndarray] = None,
                     bins: Union[int, np.ndarray, Tuple] = 100,
                     **kwargs) -> Any:
    '''
    Plot contour lines in the given axis.

    The contour intervals for the 0.68 and 0.95 confidence interval a plotted.

    :param ax: Axes to plot data in.
    :param x_value: Data on the x-axis.
    :param y_value: Data on the y-axis.
    :param limits: Limits of the input data. Shaped (2, 2) with the limits
        for the x-values in the first row and the values for the y-data
        in the second.
    :param bins: Bin argument passed to `np.histogram2d`.
    :return: Artist returned by `ax.contour`
    '''

    x_mesh, y_mesh, values = _fit_kde(x_values, y_values, limits=limits,
                                      bins=bins)

    target_levels = [0.68, 0.95]
    levels = _find_levels(values, target_levels)
    contour = ax.contour(x_mesh, y_mesh, levels.T, levels=target_levels,
                         **kwargs)
    return contour
