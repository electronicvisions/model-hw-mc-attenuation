#!/usr/bin/env python
import unittest

from model_hw_mc_attenuation.scripts.record_variations import main
from model_hw_mc_attenuation.scripts.record_variations_arbor import \
    main as main_arbor


class TestRecordTrace(unittest.TestCase):
    def test_bss(self):
        length = 4
        repetitions = 3
        result = main(length, repetitions)
        self.assertEqual(result.values.shape, (repetitions, length**2))

    def test_arbor(self):
        length = 4
        repetitions = 3
        result = main_arbor(length, repetitions)
        self.assertEqual(result.values.shape, (repetitions, length**2))


if __name__ == "__main__":
    unittest.main()
