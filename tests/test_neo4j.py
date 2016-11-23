import logging
import os
import unittest
from tempfile import TemporaryDirectory

import py2neo
from hetnetana.struct import multihetnet_examples
from hetnetana.struct.multihetnet_io import from_neo4j as mhn_from_neo4j
from hetnetana.struct.multihetnet_io import to_neo4j as mhn_to_neo4j

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

logging.getLogger("neo4j.bolt").setLevel(logging.WARNING)
logging.getLogger("httpstream").setLevel(logging.WARNING)


@unittest.skipUnless('NEO_PATH' in os.environ, 'Need NEO_PATH to configure test')
class TestNeoIo(unittest.TestCase):
    def setUp(self):
        self.standard = multihetnet_examples.generate_example_1()
        self.td = TemporaryDirectory()
        self.hetnetana_test_context = 'HETNETANA_TEST'
        self.resources = self.standard.to_resource(self.td.name)

        self.neo_graph = py2neo.Graph(os.environ['NEO_PATH'])
        self.neo_graph.run('match (n)-[r]-(v) where r.context_hetnetana = "{}" detach delete n, r, v'.format(
            self.hetnetana_test_context))

        self.qnames = {
            'g': 'Gene',
            'm': 'miRNA',
            's': 'SNP'
        }

    def doCleanups(self):
        self.neo_graph.run('match (n)-[r]-(v) where r.context_hetnetana = "{}" detach delete n, r, v'.format(
            self.hetnetana_test_context))
        self.td.cleanup()

    def test_upload(self):
        mhn_to_neo4j(self.standard, self.neo_graph, self.qnames, context=self.hetnetana_test_context)

        x = self.neo_graph.run(
            'MATCH (n) OPTIONAL MATCH (n)-[r]->(z) where r.context_hetnetana = "{}" RETURN count(r) as count'.format(
                self.hetnetana_test_context)).evaluate()
        self.assertEqual(len(self.standard.edges()), x)

        x = self.neo_graph.run(
            'MATCH (n)-[r]-(z) where r.context_hetnetana = "{}" RETURN n.name as u, z.name as v'.format(
                self.hetnetana_test_context)).data()
        x = {tuple(sorted([edge['u'], edge['v']])) for edge in x}
        st_cn = {tuple(sorted(edge)) for edge in self.standard.edges()}

        self.assertSetEqual(st_cn, x)

    def test_io(self):
        mhn_to_neo4j(self.standard, self.neo_graph, self.qnames, context=self.hetnetana_test_context)

        test = mhn_from_neo4j(self.resources, self.neo_graph, self.qnames, context=self.hetnetana_test_context)

        self.assertSetEqual(set(test.nodes()), set(self.standard.nodes()))

        test_edge_map = test.get_edge_map()
        standard_edge_map = self.standard.get_edge_map()

        self.assertSetEqual(set(test_edge_map), set(standard_edge_map))

        for u, v, k in test_edge_map:
            test_edges = test_edge_map[u, v, k]
            standard_edges = standard_edge_map[u, v, k]

            test_canonical_edges = {tuple(sorted(edge)) for edge in test_edges}
            standard_canonical_edges = {tuple(sorted(edge)) for edge in standard_edges}

            self.assertSetEqual(test_canonical_edges, standard_canonical_edges, msg='Missing {}'.format(k))

            # TODO tests annotations
