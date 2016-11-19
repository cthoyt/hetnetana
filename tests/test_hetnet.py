import os
import unittest
from tempfile import TemporaryDirectory

import pandas as pd
from scipy.stats import entropy

from hetnetana import *
from hetnetana.generation.generate import *
from hetnetana.generation.generate_toy import convert_simple_to_terminal
from hetnetana.struct import hetnet_examples
from hetnetana.struct.hetnet import decode_color_path
from tests.utils import HetnetanaTestBase

log = logging.getLogger()


class TestColorPathTransport(unittest.TestCase):
    def test_encode_error(self):
        with self.assertRaises(ValueError):
            encode_color_path('AB', color_path_type='stevecarlsberg')

    def test_malformed_error(self):
        with self.assertRaises(ValueError):
            validate_color_path('AB', 'simple', valid_colors='ACDE')
        with self.assertRaises(ValueError):
            validate_color_path(['A', ['B', []]], 'terminal', valid_colors='ACDE')
        with self.assertRaises(ValueError):
            validate_color_path([['A', []], ['B', []]], 'all', valid_colors='ACDE')

    def test_encode_simple(self):
        self.assertEqual("", encode_color_path(()))
        self.assertEqual("A", encode_color_path(('A',)))
        self.assertEqual("A", encode_color_path('A'))
        self.assertEqual("A&B", encode_color_path(('A', 'B')))
        self.assertEqual("A&B", encode_color_path('AB'))

    def test_encode_terminal(self):
        self.assertEqual("A&B|",
                         encode_color_path(('A', ('B', ())), color_path_type='terminal'))
        self.assertEqual("A&B|0:1",
                         encode_color_path(('A', ('B', ((0, 1),))), color_path_type='terminal'))
        self.assertEqual("A&B|0:1;1:0",
                         encode_color_path(('A', ('B', ((0, 1), (1, 0)))), color_path_type='terminal'))

    def test_encode_all(self):
        self.assertEqual("A|&B|",
                         encode_color_path((('A', ()), ('B', ())), color_path_type='all'))
        self.assertEqual("A|0:1&B|",
                         encode_color_path((('A', ((0, 1),)), ('B', ())), color_path_type='all'))
        self.assertEqual("A|0:1;1:0&B|",
                         encode_color_path((('A', ((0, 1), (1, 0))), ('B', ())), color_path_type='all'))

    def test_decode_error(self):
        with self.assertRaises(ValueError):
            decode_color_path('', color_path_type='steve carlsberg')

    def test_decode_simple(self):
        self.assertEqual(decode_color_path(""), ())
        self.assertEqual(decode_color_path("A"), ('A',))
        self.assertEqual(decode_color_path("A&B", ), ('A', 'B'))

    def test_decode_terminal(self):
        self.assertEqual(decode_color_path("A&B|", color_path_type='terminal'),
                         ('A', ('B', ())))
        self.assertEqual(decode_color_path("A&B|0:1", color_path_type='terminal'),
                         ('A', ('B', (('0', '1'),))))
        self.assertEqual(decode_color_path("A&B|0:1;1:0", color_path_type='terminal'),
                         ('A', ('B', (('0', '1'), ('1', '0')))))

    def test_decode_all(self):
        self.assertEqual(decode_color_path("A|&B|", color_path_type='all'),
                         (('A', ()), ('B', ())))
        self.assertEqual(decode_color_path("A|0:1&B|", color_path_type='all'),
                         (('A', (('0', '1'),)), ('B', ())))
        self.assertEqual(decode_color_path("A|0:1;1:0&B|", color_path_type='all'),
                         (('A', (('0', '1'), ('1', '0'))), ('B', ())))


class TestHetNetBase(HetnetanaTestBase):
    def setUp(self):
        self.hn = hetnet_examples.generate_example_1()


class TestHetNetTransport(TestHetNetBase):
    def test_to_resource(self):
        with TemporaryDirectory() as td:
            self.hn.to_resource(td)
            reloaded = HetNet.from_resource(os.path.join(td, 'resources.json'))
            self.assertGraphEqual(self.hn, reloaded)

    def test_to_lists(self):
        with TemporaryDirectory() as td:
            self.hn.to_lists(td, sep='\t')

            nodefilepath = os.path.join(td, 'nodelist.tsv')
            self.assertTrue(os.path.exists(nodefilepath))
            with open(nodefilepath) as nodefile:
                next(nodefile)
                for line in nodefile:
                    node, color = line.strip().split('\t')
                    self.assertIn(node, self.hn)

            edgefilepath = os.path.join(td, 'edgelist.tsv')
            self.assertTrue(os.path.exists(edgefilepath))
            edges = set(self.hn.edges())
            with open(edgefilepath) as edgefile:
                next(edgefile)
                for line in edgefile:
                    source, target = line.strip().split('\t')
                    self.assertIn((source, target), edges)


class TestHetNetProperties(TestHetNetBase):
    def test_properties(self):
        self.assertEqual(40, len(self.hn))
        self.assertEqual(40, self.hn.number_of_nodes())
        self.assertEqual(56, self.hn.number_of_edges())
        self.assertEqual({'g', 'm', 'p'}, self.hn.colors)
        self.assertIsNotNone(repr(self.hn))

    def test_get_color(self):
        self.assertEqual(hgnc.color, self.hn.get_color(hgnc(1)))
        self.assertEqual(up.color, self.hn.get_color(up(4)))
        self.assertEqual(mi.color, self.hn.get_color(mi(17)))

    def test_get_annotations(self):
        self.assertEqual('UP', self.hn.get_annotation(hgnc(1), 'regulation'))
        self.assertIsNone(self.hn.get_annotation(hgnc(1), 'MISSING'))
        self.assertEqual('SENTINEL', self.hn.get_annotation(hgnc(1), 'MISSING', default='SENTINEL'))

    def test_get_color_map(self):
        color_dict = self.hn.get_color_map()
        self.assertCountEqual(color_dict[hgnc.color], set(hgnc(i) for i in range(1, 21)))
        self.assertCountEqual(color_dict[up.color], set(up(i) for i in range(1, 17)))
        self.assertCountEqual(color_dict[mi.color], set(mi(i) for i in range(17, 21)))

    def test_get_edge_map(self):
        color_dict = self.hn.get_edge_map()
        self.assertEqual(set(color_dict[hgnc.color, mi.color]),
                         {(hgnc(17), mi(17)), (hgnc(18), mi(18)), (hgnc(19), mi(19)),
                          (hgnc(20), mi(20))})

    def test_node_matches(self):
        self.assertIn(hgnc(1), self.hn)
        self.assertTrue(self.hn.node_matches(hgnc(1), 'g', {'regulation': 'UP'}))
        self.assertFalse(self.hn.node_matches(hgnc(1), 'p', {'regulation': 'UNREGULATED'}))
        self.assertFalse(self.hn.node_matches(hgnc(1), 'p', {'regulation': 'UP'}))
        self.assertFalse(self.hn.node_matches(hgnc(1), 'p', {'regulation': 'WRONG'}))

        self.assertIn(up(4), self.hn)
        self.assertTrue(self.hn.node_matches(up(4), 'p', {}))
        self.assertFalse(self.hn.node_matches(up(4), 'p', {'regulation': 'UNREGULATED'}))


class TestColorize(TestHetNetBase):
    """COLORIZING"""

    def setUp(self):
        super().setUp()
        self.walk = hgnc(6), mi(17), up(5), hgnc(5)

    def test_colorize_simple(self):
        scp = self.hn.colorize_path(self.walk)
        self.assertEqual(list(scp), list('gmpg'))

    def test_colorize_terminal(self):
        annotations_to_study = {"g": ["regulation"], "m": ["c1", "c2"]}
        tcp = self.hn.colorize_path(self.walk, annotations=annotations_to_study, color_path_type="terminal")
        tcp_expected = ('g', 'm', 'p', ('g', (('regulation', 'UNREGULATED'),)))
        self.assertEqual(tcp, tcp_expected)

    def test_colorize_terminal_defaults(self):
        tcp = self.hn.colorize_path(self.walk, color_path_type="terminal")
        tcp_expected = ('g', 'm', 'p', ('g', (('regulation', 'UNREGULATED'),)))
        self.assertEqual(tcp, tcp_expected)

    def test_colorize_terminal_boring(self):
        tcp = self.hn.colorize_path(self.walk[:-1], color_path_type="terminal")
        tcp_expected = ('g', 'm', ('p', ()))
        self.assertEqual(tcp, tcp_expected)

    def test_colorize_all(self):
        annotations_to_study = {"g": ["regulation"], "m": ["c1", "c2"]}
        acp = self.hn.colorize_path(self.walk, annotations=annotations_to_study, color_path_type="all")
        self.assertEqual(acp, (('g', (('regulation', 'UNREGULATED'),)), ('m', (('c1', 'a'), ('c2', 'f'))), ('p', ()),
                               ('g', (('regulation', 'UNREGULATED'),))))

    def test_colorize_all_default(self):
        acp = self.hn.colorize_path(self.walk, color_path_type="all")
        self.assertEqual(acp, (('g', (('regulation', 'UNREGULATED'),)), ('m', (('c1', 'a'), ('c2', 'f'))), ('p', ()),
                               ('g', (('regulation', 'UNREGULATED'),))))

    def test_colorize_all_boring(self):
        acp = self.hn.colorize_path(self.walk[:-1], color_path_type="all")
        self.assertEqual(acp, (('g', (('regulation', 'UNREGULATED'),)), ('m', (('c1', 'a'), ('c2', 'f'))), ('p', ())))

    def test_colorize_error(self):
        with self.assertRaises(ValueError):
            self.hn.colorize_path('', color_path_type='carlos the scientist')


class TestColorPathMatching(TestHetNetBase):
    def test_match_simple(self):
        paths = self.hn.match_color_paths(hgnc(1), 'gpp')
        self.assertEqual(2, len(paths))
        self.assertIn((hgnc(1), hgnc(2), up(2), up(1)), paths)
        self.assertIn((hgnc(1), hgnc(2), up(2), up(6)), paths)

    def test_match_terminal(self):
        cp = ('g', 'p', 'p', ('g', (('regulation', 'UNREGULATED'),)))
        paths = self.hn.match_color_paths(hgnc(1), cp, color_path_type='terminal')
        self.assertEqual(1, len(paths))
        expected = (hgnc(1), hgnc(2), up(2), up(6), hgnc(6))
        self.assertEqual(expected, paths[0])

    def test_match_total(self):
        cp = (['g', [('regulation', 'UNREGULATED')]], ['p', []], ['p', []], ('g', (('regulation', 'UNREGULATED'),)))
        paths = self.hn.match_color_paths(hgnc(1), cp, color_path_type='all')
        self.assertEqual(1, len(paths))
        self.assertEqual((hgnc(1), hgnc(2), up(2), up(6), hgnc(6)), paths[0])

    def test_match_node_missing_fail(self):
        with self.assertRaises(AssertionError):
            self.hn.match_color_paths('', '')

    def test_match_fail(self):
        with self.assertRaises(ValueError):
            self.hn.match_color_paths(hgnc(1), '', color_path_type='cecil palmer')

    def test_match_paths_fail(self):
        with self.assertRaises(ValueError):
            self.hn.multi_match_color_paths(hgnc(1), color_path_type='cecil palmer')


class TestInductionFunctions(TestHetNetBase):
    """ EDGE INDUCTION """

    def setUp(self):
        super().setUp()
        self.induce_example = {(up(5), mi(20)), (up(5), mi(19)), (up(5), mi(18)), (up(5), mi(17))}
        self.start = hgnc(5)
        self.color_path = 'pm'

    def test_pairs_error(self):
        with self.assertRaises(AssertionError):
            get_inducible_pairs_indexed(self.hn, self.start, self.color_path, split=2)

    def test_pairs(self):
        """
        equivalent to terminal matching
        """
        tests = [
            get_inducible_pairs_indexed(self.hn, self.start, self.color_path, split=1),
            get_inducible_pairs_indexed(self.hn, self.start, self.color_path, split=-1),
            get_inducible_pairs_indexed(self.hn, self.start, self.color_path)
        ]

        for h, t in tests:
            self.assertEqual({up(5)}, set(h), 'head matching failed')
            self.assertEqual({mi(17), mi(18), mi(19), mi(20)}, set(t), 'tail matching failed')

    def test_no_pairs(self):
        pairs = get_inducible_pairs(self.hn, self.start, 'mmm', n_edges=1)
        self.assertCountEqual([], pairs)


class TestSimpleInduction(TestHetNetBase):
    def setUp(self):
        super().setUp()
        self.start = hgnc(5)
        self.color_path = 'pmg'

    def test_split0(self):
        h_expected, t_expected = {self.start}, {up(i) for i in (3, 5, 13, 4, 3, 9, 7, 16)}
        h, t = get_inducible_pairs_indexed(self.hn, self.start, self.color_path, split=0)

        self.assertCountEqual(h_expected, h, 'head matching failed')
        self.assertCountEqual(t_expected, t, 'tail matching failed')

    def test_split1(self):
        h_expected, t_expected = {up(5)}, {mi(17), mi(18), mi(19), mi(20)}

        h, t = get_inducible_pairs_indexed(self.hn, self.start, self.color_path, split=1)

        self.assertCountEqual(h_expected, h, 'head matching failed')
        self.assertCountEqual(t_expected, t, 'tail matching failed')

    def test_split2(self):
        h_expected, t_expected = {mi(20)}, {hgnc(i) for i in range(1, 21)}

        h, t = get_inducible_pairs_indexed(self.hn, self.start, self.color_path, split=2)

        self.assertCountEqual(h_expected, h, 'head matching failed')
        self.assertCountEqual(t_expected, t, 'tail matching failed')

    def test_total_pairs(self):
        split_0 = itt.product({self.start}, {up(i) for i in (3, 5, 13, 4, 3, 9, 7, 16)})
        split_1 = itt.product([up(5)], [mi(17), mi(18), mi(19), mi(20)])
        split_2 = itt.product([mi(20)], (hgnc(i) for i in range(1, 21)))
        expected = itt.chain(split_0, split_1, split_2)

        calculated = get_inducible_pairs_total(self.hn, self.start, self.color_path)

        self.assertCountEqual(expected, calculated)

    def test_get_induced_pairs(self):
        split_0 = itt.product({self.start}, {up(i) for i in (3, 5, 13, 4, 3, 9, 7, 16)})
        split_1 = itt.product([up(5)], [mi(17), mi(18), mi(19), mi(20)])
        split_2 = itt.product([mi(20)], (hgnc(i) for i in range(1, 21)))
        expected = itt.chain(split_0, split_1, split_2)

        calculated = get_inducible_pairs(self.hn, self.start, self.color_path, n_edges=4)
        self.assertEqual(4, len(calculated))
        self.assertLessEqual(set(calculated), set(expected))

    def test_get_induced_pairs_truncated(self):
        split_0 = itt.product({self.start}, {up(i) for i in (3, 5, 13, 4, 3, 9, 7, 16)})
        split_1 = itt.product([up(5)], [mi(17), mi(18), mi(19), mi(20)])
        split_2 = itt.product([mi(20)], (hgnc(i) for i in range(1, 21)))
        expected = itt.chain(split_0, split_1, split_2)

        calculated = get_inducible_pairs(self.hn, self.start, self.color_path, n_edges=1000)
        self.assertCountEqual(calculated, expected)


class TestTerminalInduction(TestHetNetBase):
    def setUp(self):
        super().setUp()
        self.start = hgnc(5)
        self.color_path = ('p', 'm', ('g', [['regulation', 'DOWN']]))
        self.color_path_type = 'terminal'

    def test_split0(self):
        """
        should give start node X proteins that have M&G|down
        """
        h_expected = {self.start}
        t_expected = {up(7)}

        h, t = get_inducible_pairs_indexed(self.hn, self.start, self.color_path, color_path_type='terminal', split=0)

        self.assertCountEqual(h_expected, h, 'head matching failed')
        self.assertCountEqual(t_expected, t, 'tail matching failed')

    def test_split1(self):
        """
        should give all proteins attached to start node X mirnas connected to downregulated genes
        """

        h_expected = {up(5)}
        t_expected = {mi(18)}

        h, t = get_inducible_pairs_indexed(self.hn, self.start, self.color_path, color_path_type=self.color_path_type,
                                           split=1)

        self.assertCountEqual(h_expected, h, 'head matching failed')
        self.assertCountEqual(t_expected, t, 'tail matching failed')

    def test_split2(self):
        """
        should give all mirnas connected to proteins connected to start X down regulated genes
        """

        h_expected = {mi(20)}
        t_expected = {hgnc(3), hgnc(18)}

        h, t = get_inducible_pairs_indexed(self.hn, self.start, self.color_path, color_path_type=self.color_path_type,
                                           split=2)

        self.assertCountEqual(h_expected, h, 'head matching failed')
        self.assertCountEqual(t_expected, t, 'tail matching failed')

    def test_total(self):
        """
        tests all possible splits together
        """
        split_on_0 = itt.product({self.start}, {up(7)})
        split_on_1 = itt.product({up(5)}, {mi(18)})
        split_on_2 = itt.product({mi(20)}, {hgnc(3), hgnc(18)})
        expected = itt.chain(split_on_0, split_on_1, split_on_2)

        calculated = get_inducible_pairs_total(self.hn, self.start, self.color_path,
                                               color_path_type=self.color_path_type)

        self.assertCountEqual(expected, calculated, 'missing elements')

    def test_get_induced_pairs(self):
        split_on_0 = itt.product({self.start}, {up(7)})
        split_on_1 = itt.product({up(5)}, {mi(18)})
        split_on_2 = itt.product({mi(20)}, {hgnc(3), hgnc(18)})
        expected = set(itt.chain(split_on_0, split_on_1, split_on_2))

        calculated = get_inducible_pairs(self.hn, self.start, self.color_path, color_path_type=self.color_path_type,
                                         n_edges=4)

        self.assertLessEqual(set(calculated), set(expected))


class TestHetNetAnalytical(TestHetNetBase):
    def test_get_distribution(self):
        d = get_distribution(self.hn, 'm')
        self.assertDictEqual(d, {0: 29, 1: 10, 2: 1})

    def test_get_percentile(self):
        self.assertEqual(3, get_percentile(self.hn, 'pm', 100))

    def test_get_percentile_diff(self):
        p100, p80, pdiff = get_percentile_diff(self.hn, 'pm', 100, 80)
        self.assertEqual(3, p100)
        self.assertEqual(2, p80)
        self.assertEqual(0.05, pdiff)

    def test_random_walk_l1(self):
        """
        this tests causes a length of one longer than the argument because it also contains the initial node
        """
        for _ in range(1000):
            w1 = self.hn.random_walk(hgnc(11), length=1)
            self.assertEqual(2, len(w1))

    def test_random_walk_l2(self):
        for _ in range(1000):
            w2 = self.hn.random_walk(hgnc(11), length=2)
            self.assertEqual(3, len(w2))

    def test_get_walks_exhaustive_nd_d2(self):
        walks_d2 = set(self.hn._get_walks_exhaustive(hgnc(16), length=2))
        walks_d2_expected = {
            (hgnc(16), hgnc(10), hgnc(7)),
            (hgnc(16), hgnc(10), hgnc(8)),
            (hgnc(16), hgnc(10), up(10)),
            (hgnc(16), up(16), mi(19)),
            (hgnc(16), up(16), up(11)),
        }
        self.assertEqual(walks_d2, walks_d2_expected)

    def test_get_walks_exhaustive_nd_d3(self):
        """
        commented out ones will get discarded upon seeing them since they're duplicates
        """
        walks_d3 = set(self.hn._get_walks_exhaustive(hgnc(16), length=3))
        walks_d3_expected = {
            (hgnc(16), hgnc(10), hgnc(7), up(7)),
            # (hgnc(16', hgnc(10', hgnc(7', hgnc(10'),
            (hgnc(16), hgnc(10), hgnc(8), hgnc(9)),
            (hgnc(16), hgnc(10), hgnc(8), up(8)),
            (hgnc(16), hgnc(10), hgnc(8), hgnc(4)),
            # (hgnc(16', hgnc(10', up(010', hgnc(10'),
            (hgnc(16), hgnc(10), up(10), up(8)),
            (hgnc(16), hgnc(10), up(10), up(9)),
            (hgnc(16), hgnc(10), up(10), up(7)),
            (hgnc(16), hgnc(10), up(10), up(11)),
            # (hgnc(16', up(016', mi(19', up(016),
            (hgnc(16), up(16), mi(19), up(7)),
            (hgnc(16), up(16), mi(19), hgnc(19)),
            # (hgnc(16', up(016', up(011', up(016),
            (hgnc(16), up(16), up(11), hgnc(11)),
            (hgnc(16), up(16), up(11), up(10)),
        }
        self.assertEqual(walks_d3, walks_d3_expected)

    def test_random_converges(self):
        """
        Checks that a stochastic color path retrieval converges on the exhaustive (without duplicates)
        color path retrieval
        """
        # TODO improve testing for path length > 1

        start = up(7)

        d = self.hn.get_walks(start, length=1)
        dh = Counter(path[-1] for path in d)
        dhs = sum(dh.values())
        dhn = {k: v / dhs for k, v in dh.items()}

        s = self.hn.get_walks(start, length=1, n_walks=10000, stochastic=True)
        sh = Counter(path[-1] for path in s)
        shs = sum(sh.values())
        shn = {k: v / shs for k, v in sh.items()}

        self.assertEqual(set(sh), set(dh), "stochastic search didn't reach all color paths")

        for key in sorted(sh):
            self.assertAlmostEqual(dhn[key], shn[key], delta=0.1, msg="not close enough")

    def test_stochastic_footprint(self):
        target = [up(7)]
        ef = self.hn.calculate_footprints(nodes=target, max_length=2, stochastic=False, normalize=True, n_walks=50000)
        sf = self.hn.calculate_footprints(nodes=target, max_length=2, stochastic=True, normalize=True, n_walks=50000)

        self.assertEqual(set(ef.columns), set(sf.columns))

        for i in ef.index:
            for j in ef.columns:
                self.assertAlmostEqual(ef[j][i], sf[j][i], delta=0.1)

    def test_random_walk_failure(self):
        h = HetNet({'r': {}})
        h.add_node('r1', {'color': 'r'})
        w = h.random_walk('r1', 2)
        self.assertIsNone(w)


class TestFootprint(unittest.TestCase):
    def setUp(self):
        self.hn = hetnet_examples.generate_example_2()

    def test_footprint(self):
        test_data_cols = ['r', 'b', 'g', 'r&r', 'r&b', 'r&g', 'b&b', 'b&r', 'b&g', 'g&r', 'g&b']
        test_data_idx = ['r1', 'r2', 'b1', 'b2', 'g1']
        test_data = [
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [2, 1, 0, 2, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 2, 0, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]
        ]

        test_df = pd.DataFrame(test_data, columns=test_data_cols, index=test_data_idx, dtype=float)
        df = self.hn.calculate_footprints(nodes=test_data_idx, max_length=2, color_path_type='simple')

        for i in test_data_cols:
            for j in test_data_idx:
                self.assertEqual(test_df[i][j], df[i][j])

    def test_isolated(self):
        self.hn.add_node('r_isolated', dict(color='r'))
        test_data_idx = ['r1', 'r2', 'b1', 'b2', 'g1', 'r_isolated']
        df = self.hn.calculate_footprints(nodes=test_data_idx, max_length=1, color_path_type='simple')
        for val in df.loc['r_isolated'].values:
            self.assertEqual(0, val)


class TestFootprintEntropy(unittest.TestCase):
    def setUp(self):
        self.hn = hetnet_examples.generate_example_3()

    def test_entropy(self):
        test = [
            ('r1', 'b', entropy([1, 1, 1]), 3),
            ('r1', 'bb', entropy([1]), 1),
            ('r1', 'bg', entropy([1, 2]), 3),
            ('r2', 'b', entropy([1, 1]), 2),
            ('r2', 'bb', entropy([1, 1]), 2),
            ('r2', 'bg', entropy([1, 1]), 2)
        ]

        df = self.hn.calculate_footprints(nodes=['r1', 'r2'], max_length=2, entropies=True, color_path_type='simple')

        for node, path, entropy_expected, count_expected in test:
            path_str = encode_color_path(path)

            entropy_calc = df["{}_entropy".format(path_str)][node]
            self.assertAlmostEqual(entropy_expected, entropy_calc, places=3,
                                   msg='entropies different for {} {}'.format(node, path))

            count_calculated = df["{}_count".format(path_str)][node]
            self.assertEqual(count_expected, count_calculated)


class TestGeneration(unittest.TestCase):
    def test_convert_tuple(self):
        simple = ('p', 'm')
        terminal = ('p', ('m', ()))
        result = convert_simple_to_terminal(simple)
        self.assertEqual(result, terminal)

    def test_convert_list(self):
        simple = ['p', 'm']
        terminal = ['p', ('m', ())]
        result = convert_simple_to_terminal(simple)
        self.assertEqual(result, terminal)
