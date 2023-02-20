#!/usr/bin/env python

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq

import pynn_brainscales.brainscales2 as pynn

from model_hw_mc_genetic.attenuation.bss import AttenuationExperiment as \
    AttenuationBSS
from model_hw_mc_genetic.attenuation.arbor import \
    AttenuationExperiment as AttenuationArbor

from model_hw_mc_genetic.attenuation.helper import Observation
from model_hw_mc_genetic.attenuation.helper import grid_search
from model_hw_mc_genetic.scripts.plot_grid_search import main as \
    plot_grid_search


class TestGridSearch(unittest.TestCase):
    @staticmethod
    def plotting_test(data: pd.DataFrame, suffix: str):
        '''
        Create all possible plots.

        :param data: DataFrame with results from grid search.
        :param suffix: Suffix used in file name in which the plots are saved.
        '''

        fig, axs = plt.subplots(3, tight_layout=True, figsize=(3, 7))
        plot_grid_search(axs[0], data, Observation.LENGTH_CONSTANT)
        axs[0].set_title('Length Constant')

        plot_grid_search(axs[1], data, Observation.AMPLITUDE_00)
        axs[1].set_title('AMPLITUDE')

        plot_grid_search(axs[2], data, Observation.AMPLITUDES_DISTANCE,
                         target=data['amplitudes'].mean(0))
        axs[2].set_title('Deviation Heights')

        results_folder = Path('test_results')
        results_folder.mkdir(exist_ok=True)
        fig.savefig(results_folder.joinpath(f'grid_search_{suffix}.png'))

    def test_bss(self):
        attenuation_experiment = AttenuationBSS(
            Path(pynn.helper.nightly_calib_path()))

        steps = 3
        data = grid_search(attenuation_experiment, g_leak=(0, 1022, steps),
                           g_icc=(0, 1022, steps))
        data.attrs['experiment'] = 'attenuation_bss'

        self.assertEqual(steps**2, len(data))
        self.plotting_test(data, 'bss')

    def test_arbor(self):
        attenuation_experiment = AttenuationArbor(length=4)
        g_leak_lim = attenuation_experiment.recipe.tau_mem_to_g_leak(
            [30, 2] * pq.ms)
        g_icc_lim = attenuation_experiment.recipe.tau_icc_to_g_axial(
            [10, 2] * pq.ms)

        steps = 5
        data = grid_search(attenuation_experiment,
                           g_leak=(g_leak_lim[0], g_leak_lim[1], steps),
                           g_icc=(g_icc_lim[0], g_icc_lim[1], steps))
        data.attrs['experiment'] = 'attenuation_arbor'

        self.assertEqual(steps**2, len(data))
        self.plotting_test(data, 'arbor')


if __name__ == "__main__":
    unittest.main()
