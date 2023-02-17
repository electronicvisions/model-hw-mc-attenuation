#!/usr/bin/env python

import inspect
import unittest
from pathlib import Path
from typing import Callable, Set

import matplotlib.pyplot as plt
import numpy as np

from model_hw_mc_genetic.plotting.pairplot import pairplot, create_axes_grid
from model_hw_mc_genetic.plotting import density


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

        results_folder = Path('test_results')
        results_folder.mkdir(exist_ok=True)
        fig.savefig(results_folder.joinpath('pairplot.png'))


class Generic1dTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n_samples = 200

        # Gaussian approximation expects Gaussian data
        cls.samples = np.random.RandomState(42).randn(n_samples)
        cls.limits = [np.min(cls.samples), np.max(cls.samples)]

    def setUp(self):
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close()

    @classmethod
    def generate_cases(cls):
        """
        Generate test cases for all 1-dimensional plotting functions.
        """
        for plotting_func in cls.get_plotting_funcs():
            test_method = cls.generate_test(plotting_func)
            test_method.__name__ = f"test_{plotting_func.__name__}"
            setattr(cls, test_method.__name__, test_method)

    @staticmethod
    def generate_test(plotting_func: Callable) -> Callable:
        """
        Generate a test function for running a calibration of given type with
        an algorithm of given type.

        :param plotting_func: Plotting function to be tested.
        :return: Function testing the plotting function.
        """

        def test_func(self: Generic1dTest):
            '''
            Just test that no errors occur when executed with default
            arguments.
            '''
            plotting_func(self.ax, self.samples)
            plotting_func(self.ax, self.samples, self.limits)

            results_folder = Path('test_results')
            results_folder.mkdir(exist_ok=True)
            self.fig.savefig(results_folder.joinpath(
                f'{plotting_func.__name__}.png'))

        return test_func

    @staticmethod
    def get_plotting_funcs() -> Set[Callable]:
        functions = set()
        for name, func in inspect.getmembers(density, inspect.isfunction):
            if name.startswith('plot_1d'):
                functions.add(func)
        return functions


Generic1dTest.generate_cases()


class Generic2dTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n_samples = 200

        # Gaussian approximation expects Gaussian data
        cls.x_values = np.random.RandomState(42).randn(n_samples)
        cls.y_values = np.random.RandomState(43).randn(n_samples)
        cls.limits = [[np.min(cls.x_values), np.max(cls.x_values)],
                      [np.min(cls.y_values), np.max(cls.y_values)]]

    def setUp(self):
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close()

    @classmethod
    def generate_cases(cls):
        """
        Generate test cases for all 2-dimensional plotting functions.
        """
        for plotting_func in cls.get_plotting_funcs():
            test_method = cls.generate_test(plotting_func)
            test_method.__name__ = f"test_{plotting_func.__name__}"
            setattr(cls, test_method.__name__, test_method)

    @staticmethod
    def generate_test(plotting_func: Callable) -> Callable:
        """
        Generate a test function for running a calibration of given type with
        an algorithm of given type.

        :param plotting_func: Plotting function to be tested.
        :return: Function testing the plotting function.
        """

        def test_func(self: Generic2dTest):
            '''
            Just test that no errors occur when executed with default
            arguments.
            '''
            plotting_func(self.ax, self.x_values, self.y_values)
            plotting_func(self.ax, self.x_values, self.y_values, self.limits)

            results_folder = Path('test_results')
            results_folder.mkdir(exist_ok=True)
            self.fig.savefig(results_folder.joinpath(
                f'{plotting_func.__name__}.png'))

        return test_func

    @staticmethod
    def get_plotting_funcs() -> Set[Callable]:
        functions = set()
        for name, func in inspect.getmembers(density, inspect.isfunction):
            if name.startswith('plot_2d'):
                functions.add(func)
        return functions


Generic2dTest.generate_cases()


if __name__ == "__main__":
    unittest.main()
