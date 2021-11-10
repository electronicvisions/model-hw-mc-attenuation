import unittest

from model_hw_mc_genetic.scripts.minimal_experiment import main


class TestMinimalExperiment(unittest.TestCase):
    def test_experiment(self):
        self.assertIsNone(main())
