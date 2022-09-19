#!/usr/bin/env python

import unittest

import matplotlib.pyplot as plt
import numpy as np

from model_hw_mc_genetic.plot_pairplot import create_axes_grid, pairplot


class TestPairplot(unittest.TestCase):
    def test_execution(self):
        n_params = 2
        n_samples = 50

        fig = plt.figure(figsize=(n_params * 2, n_params * 2),
                         tight_layout=True)
        base_grid = fig.add_gridspec(1)

        axs = create_axes_grid(base_grid[0], n_params)

        # Generate random data
        # choose values larger away from the interval [0, 1] to see effect
        # of empty axes.
        target_params = np.random.randint(10, 100, size=n_params)
        data = np.random.randint(-50, 50, size=(n_samples, n_params)) \
            + target_params
        labels = [f'Param {n_param}' for n_param in range(n_params)]

        artists = pairplot(axs, data, labels=labels,
                           target_params=target_params)

        self.assertEqual(artists.shape, (n_params, n_params))

        # Two pairplots in the same axis
        data = np.random.randint(-50, 50, size=(n_samples, n_params)) \
            + target_params
        labels = [f'Param {n_param}' for n_param in range(n_params)]

        artists = pairplot(axs, data, labels=labels)

        fig.savefig('test_pairplot.png')


if __name__ == "__main__":
    unittest.main()
