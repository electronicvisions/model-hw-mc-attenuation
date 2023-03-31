#!/usr/bin/env python

import unittest
from itertools import product

import numpy as np

from model_hw_mc_attenuation import fit_exponential


class TestFitLengthConstant(unittest.TestCase):
    @staticmethod
    def _get_data(location, tau, offset, scaling_factor, sigma):
        def fitfunc(location, tau, offset, scaling_factor):
            return scaling_factor * np.exp(- location / tau) + offset
        # sample from exponential
        height = fitfunc(location, tau, offset, scaling_factor)
        # add Gaussian noise
        height += np.random.normal(0, sigma, height.shape)
        return height

    def test_typical_madc_ranges(self):
        '''
        Test that fitted parameters agree with original parameters.

        We generate dummy data for different taus, scaling factors and offsets.
        Then we fit an exponential to this data and assert that we get the
        original parameters back.
        '''

        offsets = np.array([-5, 400])
        scaling_factors = np.array([100, 200])
        taus = np.array([0.75, 1.25, 1.75])

        for target_params in product(taus, offsets, scaling_factors):
            comps = np.arange(5)
            height = self._get_data(comps, *target_params, 0)
            fitted_params = fit_exponential(height)
            for target, fitted in zip(target_params, fitted_params):
                self.assertAlmostEqual(target, fitted)

    def test_with_noise(self):
        '''
        Test that fit is running when noise is added to the generated data.

        We generate dummy data for different taus, scaling factors and offsets.
        To this data noise is added and an exponential is fitted.
        We just test that the fitting does not throw any errors. We do not
        evaluate the quality of the fit.

        This aims to ensure that the fit routine can deal with the typical
        noise encountered during experiments.
        '''

        offsets = np.array([-5, 400])
        scaling_factors = np.array([100, 200])
        taus = np.array([0.75, 1.25, 1.75])

        # The normal noise level of the resting membrane is approximately
        # 3 LSB for MADC measurements.
        noise = 3

        for target_params in product(taus, offsets, scaling_factors):
            comps = np.arange(5)
            heights = self._get_data(comps, *target_params, noise)
            fit_exponential(heights)


if __name__ == "__main__":
    unittest.main()
