#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd

from model_hw_mc_genetic.plot_pairplot import pairplot, create_axes_grid
from model_hw_mc_genetic.plot_density import get_xy_1d_hist, \
    plot_1d_empty, plot_2d_empty


class ScatterPlotAnimater:
    '''
    Create a animated scatter plot of the parameter space.
    '''
    def __init__(self, data_file: Path, grid: plt.SubplotSpec,
                 step_size: int = 1):
        '''
        :param data_file: Path to the pickled DataFrame of the experiment
            result.
        :param grid: Grid position in which to create the plots.
        :param step_size: Number of generations between frames in animation.
        '''
        self.step_size = step_size
        self.data = pd.read_pickle(data_file)
        self.limits = self.data.attrs['limits']
        self.param_names = list(self.data['parameters'].columns)

        self.axes = create_axes_grid(grid, len(self.param_names))

        # Create animation
        self.artists_grid = []  # Will be set in _setup_plot
        self.generation_text = []  # Will be set in _setup_plot
        frames = len(self.data['others']['round'].unique()) // step_size
        self.animation = animation.FuncAnimation(
            grid.get_gridspec().figure, self._update, frames=frames,
            interval=100, repeat_delay=2000, repeat=False,
            init_func=self._setup_plot, blit=True)

    def _setup_plot(self):
        kwargs = {'alpha': 0.5, 's': 3}
        samples = self.data.loc[self.data['others']['round'] == 0,
                                'parameters'].values
        self.artists_grid = pairplot(self.axes, samples,
                                     labels=self.param_names,
                                     plot_1d_dist=plot_1d_empty,
                                     plot_2d_dist=plot_2d_empty,
                                     kwargs_2d=kwargs,
                                     limits=self.limits)
        ax = self.axes[-1, 0]
        self.generation_text = ax.text(0, 0, "", transform=ax.transAxes,
                                       ha="left")

        # TODO determine limits
        for n_param, _ in enumerate(self.param_names):
            self.axes[n_param, n_param].set_ylim(0, 0.2)

        return self._get_artitst()

    def _get_artitst(self):
        '''
        Filter 'None' entries from artists grid.
        '''
        artists = [artist for artist_row in self.artists_grid for artist
                   in artist_row if artist is not None]
        artists.append(self.generation_text)
        return artists

    def _update(self, frame):
        curr_round = frame * self.step_size
        n_params = len(self.param_names)
        values = self.data.loc[self.data['others']['round'] == curr_round,
                               'parameters'].values
        for row in range(n_params):
            for column in range(n_params):
                artist = self.artists_grid[row, column]
                if row == column:  # 1D histogram
                    data = values[:, row]
                    artist.set_data(*get_xy_1d_hist(data, self.limits[row, :]))
                elif column > row:  # Scatter Plot
                    artist.set_offsets(values[:, (column, row)])

        self.generation_text.set_text(f'Generation {curr_round}')

        return self._get_artitst()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot individuals in the different generations in '
                    'pairwise (always two parameters) scatter plots. The gif '
                    'shows how the population evolves over time.')
    parser.add_argument('evolution_results',
                        type=str,
                        help='Path to folder with results from the genetic '
                             'evolution.')
    args = parser.parse_args()

    fig = plt.figure(figsize=(8, 8))
    base_grid = fig.add_gridspec(1)

    animator = ScatterPlotAnimater(
        Path(args.evolution_results.joinpath('individuals.pkl')), base_grid[0])

    animator.animation.save('ea_evolution.gif')
