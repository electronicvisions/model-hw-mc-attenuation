#!/usr/bin/env python

from itertools import product
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantities as pq

import pynn_brainscales.brainscales2 as pynn

from model_hw_mc_genetic.attenuation.bss import AttenuationExperiment as \
    AttenuationBSS
from model_hw_mc_genetic.attenuation.arbor import \
    AttenuationExperiment as AttenuationArbor

from model_hw_mc_genetic.scripts.attenuation_grid_search import main
from model_hw_mc_genetic.scripts.plot_grid_search import \
    extract_observable, plot_parameter_space


class TestGridSearch(unittest.TestCase):
    @staticmethod
    def plotting_test(data: pd.DataFrame, suffix: str):
        '''
        Create all possible plots.

        :param data: DataFrame with results from grid search.
        :param suffix: Suffix used in file name in which the plots are saved.
        '''

        fig, axs = plt.subplots(3, tight_layout=True, figsize=(3, 7))
        filtered_data = extract_observable(data, 'length_constant')
        plot_parameter_space(axs[0], filtered_data)
        axs[0].set_title('Length Constant')

        filtered_data = extract_observable(data, 'height')
        plot_parameter_space(axs[1], filtered_data)
        axs[1].set_title('Height')

        filtered_data = extract_observable(data, 'deviation_heights',
                                           target=data['psp_heights'])
        plot_parameter_space(axs[2], filtered_data)
        axs[2].set_title('Deviation Heights')

        fig.savefig(f'test_grid_search_{suffix}.png', dpi=300)

    def test_bss(self):
        parameters = np.array(list(product(np.linspace(0, 1022, 3),
                                           np.linspace(0, 1022, 3))))

        attenuation_experiment = AttenuationBSS(
            Path(pynn.helper.nightly_calib_path()))

        data = main(attenuation_experiment, parameters)

        self.assertEqual(len(parameters), len(data))
        self.plotting_test(data, 'bss')

    def test_arbor(self):
        attenuation_experiment = AttenuationArbor(length=4)
        g_leak_lim = attenuation_experiment.recipe.tau_mem_to_g_leak(
            [30, 2] * pq.ms)
        g_icc_lim = attenuation_experiment.recipe.tau_icc_to_g_axial(
            [10, 2] * pq.ms)

        parameters = np.array(
            list(product(np.linspace(g_leak_lim[0], g_leak_lim[1], 5),
                         np.linspace(g_icc_lim[0], g_icc_lim[1], 5))))

        data = main(attenuation_experiment, parameters)

        self.assertEqual(len(parameters), len(data))
        self.plotting_test(data, 'arbor')


if __name__ == "__main__":
    unittest.main()
