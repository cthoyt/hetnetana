import json
import logging
import os
import tempfile
import unittest

from hetnetana.struct.multihetnet_io import from_resource as multihetnet_from_resource

log = logging.getLogger('tests')


def sort_nodes_key(hn):
    return lambda node: (hn.get_color(node), node)


class TestMark(unittest.TestCase):
    def setUp(self):
        self.f1 = [
            ('1', '2'),
            ('1', '3'),
            ('1', '4'),
            ('2', '4'),
            ('4', '5'),
            ('5', '6')
        ]

        self.f2 = [
            ('1', '2'),  # duplicate of f1
            ('2', '3'),  # overlapping nodes, but new edge
            ('6', '7'),  # one new node
            ('7', '8'),  # two new nodes, but still in main component
            ('9', '0')  # two new nodes in new component
        ]

        self.f3 = [
            'ab',
            'bc',
            'ac',
            'ad',
            'de',
            'ef'
        ]

        self.f4 = [
            'af',
            'fg',
            'dh',
            'hi',
            'hk'
        ]

        self.f0_connected = ['0', '1', '2', '5', '7']
        self.f0_isolated = ['11']
        self.f0 = self.f0_isolated + self.f0_connected

        self.f6 = [
            ('1', 'a'),
            ('2', 'b'),
            ('3', 'h'),
            ('4', 'a'),
            ('5', 'z'),
        ]

        self.resources = {
            'params': {
                'letter': {},
                'number': {
                    'tag': {
                        'options': ['T', 'F'],
                        'default': 'F'
                    }
                }
            },
            'edge_files': [
                {
                    'path': '1.csv',
                    'type': 'alpha',
                    'indexes': {
                        'number': [0, 1]
                    }
                },
                {
                    'path': '2.csv',
                    'type': 'beta',
                    'indexes': {
                        'number': [0, 1]
                    }
                },
                {
                    'path': '3.csv',
                    'type': 'gamma',
                    'indexes': {
                        'letter': [0, 1]
                    }
                },
                {
                    'path': '4.csv',
                    'type': 'delta',
                    'indexes': {
                        'letter': [0, 1]
                    }
                },
                {
                    'path': '5.csv',
                    'type': 'epsilon',
                    'indexes': {
                        'number': [0],
                        'letter': [1]
                    }
                }
            ],
            'node_files': [
                {
                    'path': '0.csv',
                    'color': 'number',
                    'delimiter': '\t',
                    'method': 'mark_inclusive',
                    'annotation': 'tag',
                    'value': 'T'
                }
            ]
        }

    def test_stuff(self):
        with tempfile.TemporaryDirectory() as d:
            log.debug("Temporary Directory: {}".format(d))

            for i, data in enumerate((self.f1, self.f2, self.f3, self.f4, self.f6), start=1):
                with open(os.path.join(d, "{}.csv".format(i)), 'w') as f:
                    print('source', 'target', file=f)
                    for line in data:
                        print(*line, sep='\t', file=f)

            with open(os.path.join(d, '0.csv'), 'w') as f:
                print('header', file=f)
                for node in self.f0:
                    print(node, file=f)

            self.resources['base_path'] = d
            with open(os.path.join(d, 'resources.json'), 'w') as f:
                json.dump(self.resources, f, indent=2)

            log.info('Loading MultiHetnet')
            hn = multihetnet_from_resource(os.path.join(d, 'resources.json'))

        log.info("Loaded MultiHetnet Nodes")
        for node in sorted(hn, key=sort_nodes_key(hn)):
            log.debug("{}: {}".format(node, hn.node[node]))

        for n in sorted(hn, key=sort_nodes_key(hn)):
            if hn.get_color(n) != 'number':
                continue

            self.assertNotIn(n, self.f0_isolated)

            if n in self.f0_connected:
                self.assertEqual(hn.get_annotation(n, 'tag'), 'T', msg="Failed on {}: {}".format(n, hn.node[n]))
            else:
                self.assertEqual(hn.get_annotation(n, 'tag'), 'F', msg="Failed on {}: {}".format(n, hn.node[n]))
