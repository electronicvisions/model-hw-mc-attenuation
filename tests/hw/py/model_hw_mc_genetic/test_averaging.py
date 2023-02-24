#!/usr/bin/env python
import unittest
import numpy as np

import pynn_brainscales.brainscales2 as pynn

from model_hw_mc_genetic.attenuation import extract_psp_heights
from model_hw_mc_genetic.attenuation.bss import AttenuationExperiment


class TestAveraging(unittest.TestCase):
    def test_averaging(self):
        length = 4
        n_average = 7
        experiment = AttenuationExperiment(
            pynn.helper.nightly_calib_path(), length, n_average=n_average)
        experiment.set_parameters(np.array([100, 100]))
        traces = experiment.record_membrane_traces()
        for trace in traces:
            input_spikes = trace.annotations["input_spikes"]
            self.assertLess(trace.t_stop, 1.5 * input_spikes[-1])
            self.assertEqual(len(input_spikes), length)
        psp_heights = extract_psp_heights(traces)
        self.assertEqual(len(psp_heights.flatten()), length ** 2)


if __name__ == "__main__":
    unittest.main()
