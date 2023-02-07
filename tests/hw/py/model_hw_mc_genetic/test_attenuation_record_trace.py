#!/usr/bin/env python
import unittest

from model_hw_mc_genetic.scripts.attenuation_record_trace import main
from model_hw_mc_genetic.scripts.attenuation_record_trace_arbor import main \
    as main_arbor


class TestRecordTrace(unittest.TestCase):
    def test_bss(self):
        length = 4
        result = main(length)
        self.assertEqual(len(result.segments[-1].irregularlysampledsignals),
                         length)

    def test_arbor(self):
        length = 4
        result = main_arbor(length)
        self.assertEqual(len(result.segments[-1].irregularlysampledsignals),
                         length)


if __name__ == "__main__":
    unittest.main()
