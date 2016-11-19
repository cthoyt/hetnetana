import logging
import os
import unittest
from tempfile import TemporaryDirectory

import pandas as pd

from hetnetana import *
from hetnetana.struct import multihetnet_examples
from hetnetana.struct.multihetnet import encode_color_path
from hetnetana.struct.multihetnet_io import from_resource as multihetnet_from_resource
from tests.utils import HetnetanaTestBase

log = logging.getLogger()


class TestMultiHetNetBase(HetnetanaTestBase):
    def setUp(self):
        self.hn = multihetnet_examples.generate_example_1()


class TestProperties(TestMultiHetNetBase):
    def test_stochastic(self):
        with self.assertRaises(NotImplementedError):
            self.hn.get_walks(node='bender', length=42, stochastic=True)

    def test_io(self):
        with TemporaryDirectory() as td:
            self.hn.to_resource(td)
            reloaded = multihetnet_from_resource(os.path.join(td, 'resources.json'))
            self.assertCountEqual(self.hn.nodes(), reloaded.nodes())
            self.assertGraphEqual(self.hn, reloaded)


class TestSimpleCpPathMatching(TestMultiHetNetBase):
    def test_match_simple(self):
        target_start = hgnc(1)
        target_cp = ['coexp', hgnc.color]
        paths = self.hn.match_simple_cp_path(target_start, target_cp)
        paths = list(paths)

        answers = [
            (hgnc(1), hgnc(2))
        ]

        self.assertEqual(len(answers), len(paths))
        for answer in answers:
            self.assertIn(answer, paths)

    def test_match_simple2(self):
        target_start = hgnc(1)
        target_cp = ['coexp', hgnc.color, 'ppi', hgnc.color]
        paths = self.hn.match_simple_cp_path(target_start, target_cp)
        paths = list(paths)

        answers = [
            (hgnc(1), hgnc(2), hgnc(6))
        ]

        self.assertEqual(len(answers), len(paths))

        for answer in answers:
            self.assertIn(answer, paths)


class TestTerminalCpPathMatching(TestMultiHetNetBase):
    def test_match_terminal(self):
        target_start = hgnc(1)
        target_cp = ['coexp', hgnc.color, 'ppi', [hgnc.color, [['regulation', 'UNREGULATED']]]]

        paths = self.hn.match_terminal_cp_path(target_start, target_cp)

        answers = [
            (hgnc(1), hgnc(2), hgnc(6)),
        ]

        self.assertEqual(len(answers), len(paths))
        for answer in answers:
            self.assertIn(answer, paths)

    def test_match_terminal2(self):
        target_start = hgnc(1)
        target_cp = [
            'coexp',
            hgnc.color,
            'ppi',
            hgnc.color,
            'ppi',
            [hgnc.color, [['regulation', 'UNREGULATED']]]
        ]

        paths = self.hn.match_terminal_cp_path(target_start, target_cp)

        answers = [
            (hgnc(1), hgnc(2), hgnc(6), hgnc(5)),
            (hgnc(1), hgnc(2), hgnc(6), hgnc(12)),
        ]

        self.assertEqual(len(answers), len(paths))
        for answer in answers:
            self.assertIn(answer, paths)


class TestFootprint(unittest.TestCase):
    def setUp(self):
        self.hn = multihetnet_examples.generate_example_2()

    def test_footprint(self):
        test_data_cols = [
            encode_color_path(['coexp', hgnc.color]),
            encode_color_path(['ppi', hgnc.color]),
            encode_color_path(['mti', mi.color]),
            encode_color_path(['mti', hgnc.color]),
            encode_color_path(['coexp', hgnc.color, 'mti', mi.color]),
            encode_color_path(['ppi', hgnc.color, 'mti', mi.color]),
            encode_color_path(['mti', hgnc.color, 'coexp', hgnc.color]),
            encode_color_path(['mti', hgnc.color, 'ppi', hgnc.color]),
            encode_color_path(['mti', hgnc.color, 'mti', mi.color])
        ]
        test_data_idx = [hgnc(1), hgnc(2), mi(1), mi(2), mi(3)]
        test_data = [
            [1, 1, 2, 0, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 0]
        ]

        test_df = pd.DataFrame(test_data, columns=test_data_cols, index=test_data_idx, dtype=float)
        df = self.hn.calculate_footprints(nodes=test_data_idx, max_length=2, color_path_type='simple')

        for i in test_data_cols:
            for j in test_data_idx:
                self.assertEqual(test_df[i][j], df[i][j])

    def test_isolated(self):
        self.hn.add_node(hgnc(3), dict(color=hgnc.color))
        test_data_idx = [hgnc(1), hgnc(2), mi(1), mi(2), mi(3), hgnc(3)]
        df = self.hn.calculate_footprints(nodes=test_data_idx, max_length=2, color_path_type='simple')
        for val in df.loc[hgnc(3)].values:
            self.assertEqual(0, val)
