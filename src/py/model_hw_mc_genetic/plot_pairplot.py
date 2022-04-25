from typing import Optional, List, Callable, Tuple, Any, Dict
import matplotlib.pyplot as plt
import numpy as np

from model_hw_mc_genetic.plot_density import plot_1d_hist, plot_2d_hist


def create_axes_grid(base_gs: plt.SubplotSpec, size: Tuple[int]) -> np.ndarray:
    '''
    Create a quadratic grid of axes.

    :param base_gs: Gridspec to which the axes are added.
    :param size: Number of axes per side -> in total size x size axes are
        created.
    :return: Array with the created axes.
    '''
    fig = base_gs.get_gridspec().figure
    grid = base_gs.subgridspec(size[0], size[1])
    axes = []
    for row in range(size[0]):
        axes_row = []
        for column in range(size[1]):
            axes_row.append(fig.add_subplot(grid[row, column]))
        axes.append(axes_row)
    return np.asarray(axes)


def pairplot(axes: np.ndarray,
             samples: np.ndarray, *,
             plot_1d_dist: Callable = plot_1d_hist,
             plot_2d_dist: Callable = plot_2d_hist,
             kwargs_1d: Optional[Dict[str, Any]] = None,
             kwargs_2d: Optional[Dict[str, Any]] = None,
             labels: Optional[List[str]] = None,
             limits: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Plot 1D and pairwise 2D distribution of parameters for the given samples.

    The plot is organized in a quadratic grid. On the diagonal the distribution
    of single parameters is displayed, above the diagonal the pairwise
    distribution of two parameters.

    :param axes: Quadratic array with axes to plot data in.
    :param samples: Samples to display in the plot.
    :param plot_1d_dist: Function used to plot the 1-dimensional distributions.
    :param plot_2d_dist: Function used to plot the 2-dimensional distributions.
    :param kwargs_1d: Keyword arguments supplied to plot_1d_dist.
    :param kwargs_2d: Keyword arguments supplied to plot_2d_dist.
    :param labels: Labels for the parameters. List should have the same length
        as the number of parameters.
    :param limits: Lower and upper limits for the different parameters. The
        array should have the shape (number of parameters, 2).
    :return: Grid of artists which are returned when the plots were created.
    '''

    samples = samples.reshape((-1, samples.shape[-1]))
    n_parameters = samples.shape[-1]

    artists = []
    for row in range(n_parameters):
        artists_row = []
        for column in range(n_parameters):
            ax = axes[row, column]
            if row == column:  # distribution of single parameter
                restricted_limits = None
                if limits is not None:
                    restricted_limits = limits[row, :]

                artists_row.append(plot_1d_dist(ax, samples[:, row],
                                                limits=restricted_limits,
                                                kwargs=kwargs_1d))
                if labels is not None:
                    ax.set_xlabel(labels[row])

            elif column > row:  # 2D distributions
                restricted_limits = None
                if limits is not None:
                    restricted_limits = limits[(column, row), :]

                artists_row.append(plot_2d_dist(ax,
                                                samples[:, column],
                                                samples[:, row],
                                                limits=restricted_limits,
                                                kwargs=kwargs_2d))
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            else:  # Hide unused axes
                artists_row.append(None)
                ax.set_axis_off()
        artists.append(artists_row)
    return np.array(artists)


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
