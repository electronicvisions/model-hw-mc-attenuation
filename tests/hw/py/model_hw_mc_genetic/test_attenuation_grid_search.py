#!/usr/bin/env python

from itertools import product
import unittest

import matplotlib.pyplot as plt
import numpy as np

import pynn_brainscales.brainscales2 as pynn

from model_hw_mc_genetic.scripts.chain_attenuation_grid_search import main
from model_hw_mc_genetic.scripts.plot_grid_search_result import \
    extract_observable, plot_parameter_space


class TestGridSearch(unittest.TestCase):
    def test_attenuation_experiment(self):
        parameters = np.array(list(product(np.linspace(0, 1022, 3),
                                           np.linspace(0, 1022, 3))))

        # Experiment execution
        data = main(pynn.helper.nightly_calib_path(), parameters)

        self.assertEqual(len(parameters), len(data))

        # Plotting
        fig, axis = plt.subplots()
        filtered_data = extract_observable(data, 'length_constant')
        plot_parameter_space(axis, filtered_data)
        fig.savefig('test_grid_search_length_constant.png', dpi=300)

        fig, axis = plt.subplots()
        filtered_data = extract_observable(data, 'height')
        plot_parameter_space(axis, filtered_data)
        fig.savefig('test_grid_search_height.png', dpi=300)

        fig, axis = plt.subplots()
        filtered_data = extract_observable(data, 'deviation_heights',
                                           target=data['psp_heights'])
        plot_parameter_space(axis, filtered_data)
        fig.savefig('test_grid_search_deviation_heights.png', dpi=300)


if __name__ == "__main__":
    unittest.main()
