#!/usr/bin/env python

import unittest
from pathlib import Path

import numpy as np

from model_hw_mc_genetic.abc import Algorithm
from model_hw_mc_genetic.attenuation import Observation
from model_hw_mc_genetic.attenuation.helper import get_experiment

from model_hw_mc_genetic.scripts.attenuation_record_variations_arbor import \
    main as record_variations
from model_hw_mc_genetic.scripts.attenuation_abc import main as abc
from model_hw_mc_genetic.scripts.abc_draw_posterior_samples import draw_samples
from model_hw_mc_genetic.scripts.plot_abc_pairplot import plot_pairplot
from model_hw_mc_genetic.scripts.plot_attenuation_pairplot_and_trace import \
    plot_pairplot_and_trace, plot_trace_attenuation


class TestAlgorithms(unittest.TestCase):
    '''
    Test possible ABC algorithms.

    Test all possible ABC algorithms with the observation
    :class:`Observation.LENGTH_CONSTANT` as this observation needs the lowest
    number of simulations.
    '''

    observation = Observation.LENGTH_CONSTANT

    @classmethod
    def setUpClass(cls):
        cls.target_df = record_variations(length=4, repetitions=2)

    def test_snpe(self):
        simulations = [50, 50]
        samples, posteriors = abc(self.target_df,
                                  self.observation,
                                  Algorithm.SNPE,
                                  simulations=simulations,
                                  global_parameters=True)

        self.assertEqual(len(samples), np.sum(simulations))
        self.assertEqual(len(posteriors), len(simulations))

    def test_snre(self):
        simulations = [50, 50]
        samples, posteriors = abc(self.target_df,
                                  self.observation,
                                  Algorithm.SNRE,
                                  simulations=simulations,
                                  global_parameters=True)

        self.assertEqual(len(samples), np.sum(simulations))
        self.assertEqual(len(posteriors), len(simulations))

    def test_mcabc(self):
        simulations = [200]
        samples, posteriors = abc(self.target_df,
                                  self.observation,
                                  Algorithm.MCABC,
                                  simulations=simulations,
                                  global_parameters=True)

        self.assertGreater(len(samples), 0)
        self.assertEqual(len(posteriors), 0)


class TestObservations(unittest.TestCase):
    '''
    Test possible types of observations with the SNPE algorithm.

    Only use the SNPE algorithm to save time. The functionality of the
    different algorithms is tested in :class:`TestAlgorithms`. There the
    observations :class:`Observation.LENGTH_CONSTANT` is used. This observation
    will be omitted here.
    '''

    algorithm = Algorithm.SNPE

    @classmethod
    def setUpClass(cls):
        cls.target_df = record_variations(length=4, repetitions=2)

    def test_amplitudes(self):
        simulations = [200, 50]
        samples, posteriors = abc(self.target_df,
                                  Observation.AMPLITUDES,
                                  Algorithm.SNPE,
                                  simulations=simulations,
                                  global_parameters=True)

        self.assertEqual(len(samples), np.sum(simulations))
        self.assertEqual(len(posteriors), len(simulations))

    def test_amplitudes_first(self):
        simulations = [200, 50]
        samples, posteriors = abc(self.target_df,
                                  Observation.AMPLITUDES_FIRST,
                                  Algorithm.SNPE,
                                  simulations=simulations,
                                  global_parameters=True)

        self.assertEqual(len(samples), np.sum(simulations))
        self.assertEqual(len(posteriors), len(simulations))


class TestEvaluation(unittest.TestCase):
    '''
    Test routines which are used to evaluate approximated posteriors.
    '''

    @classmethod
    def setUpClass(cls):
        cls.results_folder = Path('test_results')
        cls.results_folder.mkdir(exist_ok=True)

        target_file = cls.results_folder.joinpath('target_df.pkl')
        cls.target_df = record_variations(length=4, repetitions=2)
        cls.target_df.to_pickle(target_file)

        cls.abc_samples, cls.posteriors = abc(cls.target_df,
                                              Observation.LENGTH_CONSTANT,
                                              Algorithm.SNPE,
                                              simulations=[50, 50],
                                              global_parameters=True)
        cls.abc_samples.attrs['target_file'] = str(target_file.resolve())

        cls.posterior_samples = None

    def test_00_drawing_samples(self):
        self.__class__.posterior_samples = draw_samples(self.posteriors,
                                                        self.abc_samples)

    def test_01_plot_pairplot(self):
        figure = plot_pairplot([self.posterior_samples])
        figure.savefig(self.results_folder.joinpath('test_pairplot.png'))

    # pylint: disable=invalid-name
    def test_01_plot_pairplot_and_trace(self):
        attenuation_exp = get_experiment(self.target_df)
        params = self.target_df.attrs['parameters']

        figure = plot_pairplot_and_trace(self.posterior_samples,
                                         attenuation_exp,
                                         plot_trace_attenuation,
                                         original_parameters=params,
                                         n_samples=8)
        figure.savefig(
            self.results_folder.joinpath('test_pairplot_and_traces.png'))


if __name__ == "__main__":
    unittest.main()
