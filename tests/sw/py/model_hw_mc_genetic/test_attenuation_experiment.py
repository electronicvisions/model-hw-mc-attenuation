#!/usr/bin/env python

import os
import unittest
from unittest import mock

import pynn_brainscales.brainscales2 as pynn

from model_hw_mc_genetic.attenuation.bss import AttenuationExperiment as \
    AttenuationBSS
from model_hw_mc_genetic.attenuation.arbor import AttenuationExperiment as \
    AttenuationArbor


class TestBSS(unittest.TestCase):

    @mock.patch.dict(os.environ, {"HXCOMM_ENABLE_ZERO_MOCK": "1"}, clear=True)
    def test_creation(self):
        length = 5
        experiment = AttenuationBSS(pynn.helper.nightly_calib_path(),
                                    length=length)

        self.assertEqual(len(experiment.spike_times), length)


class TestArbor(unittest.TestCase):
    def test_creation(self):
        length = 5
        experiment = AttenuationArbor(length=length)
        self.assertEqual(len(experiment.spike_times), length)

    def test_measurement(self):
        length = 5
        experiment = AttenuationArbor(length=length)

        data = experiment.measure_response()

        self.assertEqual(data.shape, (length, length))


if __name__ == "__main__":
    unittest.main()
