#!/usr/bin/env python

import os
import unittest
from unittest import mock

import pynn_brainscales.brainscales2 as pynn

from model_hw_mc_genetic.attenuation.bss import AttenuationExperiment


class TestBSS(unittest.TestCase):

    @mock.patch.dict(os.environ, {"HXCOMM_ENABLE_ZERO_MOCK": "1"}, clear=True)
    def test_creation(self):
        length = 5
        experiment = AttenuationExperiment(pynn.helper.nightly_calib_path(),
                                           length=length)

        self.assertEqual(len(experiment.spike_times), length)


if __name__ == "__main__":
    unittest.main()
