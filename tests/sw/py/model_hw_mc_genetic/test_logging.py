#!/usr/bin/env python

import unittest

import numpy as np

from model_hw_mc_genetic.logging import ExperimentLogger


class TestLogging(unittest.TestCase):

    def test_logging(self):
        length = 20
        to_be_logged = {'parameters': ['param_0', 'param_1'],
                        'observables': ['obs_0', 'obs_1', 'obs_2'],
                        'meta_data': ['meta_0']}
        logger = ExperimentLogger(**to_be_logged)

        values = np.random.randint(100, size=(length, 6))

        for row in values:
            logger.log_statistics(row[0:2], row[2:5], row[5:6])

        data = logger.get_data_frame()

        self.assertEqual(len(data), length)

        self.assertEqual(data['parameters'].columns.to_list(),
                         to_be_logged['parameters'])
        self.assertTrue(np.all(data['parameters'].values == values[:, 0:2]))

        self.assertEqual(data['observables'].columns.to_list(),
                         to_be_logged['observables'])
        self.assertTrue(np.all(data['observables'].values == values[:, 2:5]))

        self.assertEqual(data['meta_data'].columns.to_list(),
                         to_be_logged['meta_data'])
        self.assertTrue(np.all(data['meta_data'].values == values[:, 5:6]))


if __name__ == "__main__":
    unittest.main()
