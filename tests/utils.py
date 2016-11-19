import unittest


class HetnetanaTestBase(unittest.TestCase):
    def assertGraphEqual(self, graph_a, graph_b):
        """

        :param graph_a:
        :type graph_a: nx.Graph
        :param graph_b:
        :type grapb_b: nx.Graph
        """

        self.assertEqual(set(graph_a.nodes()), set(graph_b.nodes()))
        self.assertEqual(tuple(sorted(sorted(e) for e in graph_a.edges())),
                         tuple(sorted(sorted(e) for e in graph_b.edges())))
