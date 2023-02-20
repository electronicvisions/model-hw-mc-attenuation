from typing import Optional, List, Callable, Any, Dict
import matplotlib.pyplot as plt
import numpy as np

from model_hw_mc_genetic.plotting.density import plot_1d_hist, plot_2d_scatter


def create_axes_grid(base_gs: plt.SubplotSpec, n_axes: int,
                     size_factor: int = 5) -> np.ndarray:
    '''
    Create a quadratic grid of axes.

    The size of the axes at the top and right edge are smaller since it is
    assumed that density plots are illustrated there, while the other axes
    are used to display two-dimensional parameter distributions.

    :param base_gs: Gridspec to which the axes are added.
    :param n_axes: Number of axes per side.
    :param size_factor: Factor how much bigger axes in the center
        are compared to axes at the right/top edge.
    :return: Array with the created axes.
    '''
    fig = base_gs.get_gridspec().figure

    width_ratios = [size_factor] * (n_axes - 1) + [1]
    height_ratios = [1] + [size_factor] * (n_axes - 1)
    grid = base_gs.subgridspec(n_axes, n_axes, width_ratios=width_ratios,
                               height_ratios=height_ratios, wspace=0.1,
                               hspace=0.1)
    axes = []
    for row in range(n_axes):
        axes_row = []
        for column in range(n_axes):
            axes_row.append(fig.add_subplot(grid[row, column]))
        axes.append(axes_row)
    return np.asarray(axes)


def swap_xy_data(ax):
    '''
    Swap the x- and y-data of a line plot in the givena xis.

    The data of the last line plot in the axes is swapped. The
    x- and y-limits are recalculated.

    :param ax: Axis for which the data is swapped.
    '''
    line = ax.lines[-1]

    # swap data
    newx = line.get_ydata()
    newy = line.get_xdata()

    line.set_xdata(newx)
    line.set_ydata(newy)

    # recalculate data limits and autoscale plot
    ax.relim()
    ax.autoscale()


def plot_1d_marginals(axes: List,
                      samples: np.ndarray, *,
                      plot_func: Callable = plot_1d_hist,
                      limits: Optional[np.ndarray] = None,
                      **kwargs) -> List:
    '''
    Plot 1D distribution of parameters for the given samples.

    :param axes: List of axes in which to plot the distributions.
    :param samples: Samples to display in the plot.
    :param plot_func: Function used to plot the 1-dimensional distributions.
    :param limits: Lower and upper limits for the different parameters. The
        array should have the shape (number of parameters, 2).
    :return: List of artists which are returned when the plots were created.
    '''
    artists = []
    for n_param, ax in enumerate(axes):
        restricted_limits = None
        if limits is not None:
            restricted_limits = limits[n_param, :]

        artists.append(plot_func(ax, samples[:, n_param],
                                 limits=restricted_limits,
                                 **kwargs))
    return artists


def plot_2d_marginals(axes: np.ndarray,
                      samples: np.ndarray, *,
                      plot_func: Callable = plot_1d_hist,
                      limits: Optional[np.ndarray] = None,
                      **kwargs) -> np.ndarray:
    '''
    Plot pairwise 2d marginals of the given parameters.

    On the different rows the parameters [N, 0:N-2] are displayed, in the
    different columns the parameters [0:N-1]. The distributions illustrate
    the 2d marginals of the given samples. Only in the upper triangular
    of the grid, data is plotted.

    :param axes: Grid of axes in which to plot the distributions.
    :param samples: Samples to display in the plot.
    :param plot_func: Function used to plot the 2-dimensional distributions.
    :param limits: Lower and upper limits for the different parameters. The
        array should have the shape (number of parameters, 2).
    :return: Grid of artists which are returned when the plots were created.
    '''
    n_parameters = samples.shape[1]
    artists = []
    for row in range(n_parameters - 1):
        artists_row = []
        for column in range(n_parameters - 1):
            column_param = column
            row_param = np.roll(np.arange(n_parameters), 1)[row]
            ax = axes[row, column]
            if row <= column:
                restricted_limits = None
                if limits is not None:
                    restricted_limits = limits[(column_param, row_param), :]

                artists_row.append(plot_func(ax,
                                             samples[:, column_param],
                                             samples[:, row_param],
                                             limits=restricted_limits,
                                             **kwargs))

            else:  # Hide unused axes
                artists_row.append(None)
                ax.set_axis_off()
        artists.append(artists_row)

    return np.array(artists, dtype=object)


def _set_prob_limits_1d_marginals(axes: np.ndarray):
    '''
    Determine limits of probability axis of all 1d marginal plots and set all
    limits to a common value.

    The marginals are assumed to be at the top and at the left, the upper right
    corner is assumed to be empty. On the left side the probabilities are
    assumed to be plottend on the x-axis. The limits are infered from the
    data limits.

    :pram axes: Grid of all axes already filled with data.
    '''
    limits = np.vstack([
        [(ax.dataLim.ymin, ax.dataLim.ymax) for ax in axes[0, :-1]],
        [(ax.dataLim.xmin, ax.dataLim.xmax) for ax in axes[1:, -1]]])
    new_limits = [np.min(limits), np.max(limits)]

    for ax in axes[0, :-1]:
        ax.set_ylim(new_limits)
    for ax in axes[1:, -1]:
        ax.set_xlim(new_limits)


def _mark_targets(axes: np.ndarray, targets: List[float], **kwargs):
    '''
    Mark target parameters with vertcial/horizontal lines.

    :pram axes: Grid of all axes.
    :pram target: Targets for the differnt parameters.
    '''
    # first row
    for ax, target in zip(axes[0, :-1], targets[1:]):
        ax.axvline(target, **kwargs)

    # right column
    for ax, target in zip(axes[1:, -1], targets[:-1]):
        ax.axhline(target, **kwargs)

    # just look at lower left quadrant with pariplots
    for row, axs_row in enumerate(axes[1:, :-1]):
        for column, ax in enumerate(axs_row):
            if row > column:
                # no pairplots in lower left triangular space
                continue
            column_param = column
            row_param = np.roll(np.arange(len(targets)), 1)[row]

            ax.axvline(targets[column_param], **kwargs)
            ax.axhline(targets[row_param], **kwargs)


def _style_pairplot(axes: np.ndarray):
    '''
    Remove unwanted tick labels and splines.

    Remove x- and y-ticklabels for all but the lowest diagonal of pairplots.
    For 1d marginals remove all splines but the spline where the parameters
    are drawn.

    :pram axes: Grid of all axes.
    '''

    # Disable all but spine with parameter-axis for 1d marginals
    for ax in axes[0, :-1]:
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels([])
        ax.get_yaxis().set_visible(False)

    for ax in axes[1:, -1]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticklabels([])
        ax.get_xaxis().set_visible(False)

    # remove tick labels above diagonal for pairplots
    if len(axes) <= 2:
        # just a single pairplot -> do not remove tick labels
        return
    for row, row_axes in enumerate(axes[1:, :-1]):
        for col, ax in enumerate(row_axes):
            if row > col:
                continue
            ax.set_xticklabels([])
            ax.set_yticklabels([])


def pairplot(axes: np.ndarray,
             samples: np.ndarray, *,
             plot_1d_dist: Callable = plot_1d_hist,
             plot_2d_dist: Callable = plot_2d_scatter,
             kwargs_1d: Optional[Dict[str, Any]] = None,
             kwargs_2d: Optional[Dict[str, Any]] = None,
             labels: Optional[List[str]] = None,
             target_params: Optional[List[float]] = None,
             limits: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Plot 1D and pairwise 2D distribution of parameters for the given samples.

    The plot is organized in a quadratic grid. On the top and the left
    the one-dimensional marginals are displayed. In the other axes of the upper
    triangular the two-dimensional marginals are displayed.

    :param axes: Quadratic array with axes to plot data in.
    :param samples: Samples to display in the plot.
    :param plot_1d_dist: Function used to plot the 1-dimensional distributions.
    :param plot_2d_dist: Function used to plot the 2-dimensional distributions.
    :param kwargs_1d: Keyword arguments supplied to plot_1d_dist.
    :param kwargs_2d: Keyword arguments supplied to plot_2d_dist.
    :param labels: Labels for the parameters. List should have the same length
        as the number of parameters.
    :pram target_parmas: Targets for the differnt parameters.
    :param limits: Lower and upper limits for the different parameters. The
        array should have the shape (number of parameters, 2).
    :return: Grid of artists which are returned when the plots were created.
    '''
    if kwargs_1d is None:
        kwargs_1d = {}

    if kwargs_2d is None:
        kwargs_2d = {}

    if limits is None:
        limits = np.vstack([samples.min(0), samples.max(0)]).T

    n_parameters = samples.shape[-1]
    artists = np.zeros((n_parameters, n_parameters), dtype=object)

    # 1d marginals at the top
    artists[0, :-1] = plot_1d_marginals(axes[0, :-1],
                                        samples[:, :-1],
                                        plot_func=plot_1d_dist,
                                        limits=limits[:-1],
                                        **kwargs_1d)

    # 1d marginals at the right
    artists[1:, -1] = plot_1d_marginals(axes[1:, -1],
                                        np.roll(samples, 1)[:, :-1],
                                        plot_func=plot_1d_dist,
                                        limits=np.roll(limits, 1, axis=0)[:-1],
                                        **kwargs_1d)

    for ax in axes[1:, -1]:
        swap_xy_data(ax)

    # 2d marginals
    artists[1:, :-1] = plot_2d_marginals(axes[1:, :-1], samples,
                                         plot_func=plot_2d_dist,
                                         limits=limits,
                                         **kwargs_2d)

    # add labels to pariplots
    if labels is not None:
        for n_diagonal, ax in enumerate(axes[1:, :-1].diagonal()):
            ax.set_xlabel(labels[n_diagonal])
            ax.set_ylabel(np.roll(labels, 1)[n_diagonal])

    # x/y-limits of pairplot is trivally shared between columns/rows.
    # share these limits with 1d marginals
    for n_axs, ax in enumerate(axes[0, :-1]):
        ax.set_xlim(axes[1, n_axs].get_xlim())
    for n_axs, ax in enumerate(axes[1:, -1], start=1):
        ax.set_ylim(axes[n_axs, -2].get_ylim())

    if target_params is not None:
        _mark_targets(axes, target_params, alpha=0.3, color='k', lw=0.5)

    _set_prob_limits_1d_marginals(axes)
    _style_pairplot(axes)
    axes[0, -1].set_axis_off()

    return artists


def annotate_samples_pairplot(axes: np.ndarray, samples: np.ndarray,
                              labels: Optional[List[str]] = None) -> None:
    '''
    Annotate samples in a pairplot.

    :param axes: Quadratic array with axes in which to annotate the samples.
    :param samples: Parameters of samples to annotate.
    :param labels: Labels to use for the annotation. Should have the same
        length as `samples`. If not supplied the samples are enumerated in an
        increasing fashion.
    '''

    samples = samples.reshape((-1, samples.shape[-1]))
    if labels is None:
        labels = np.arange(len(samples))

    n_parameters = samples.shape[-1]

    for row in range(n_parameters):
        for column in range(n_parameters):
            ax = axes[row, column]
            if column > row:  # 2D distributions
                for label, params in zip(labels, samples):
                    ax.annotate(label, (params[column], params[row]))
