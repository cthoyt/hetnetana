from itertools import combinations, product, accumulate

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import uniform


class HBNetwork(nx.Graph):
    """ Old method for generating heterogeneous binomial networks
    """

    def __repr__(self):
        return "HBNetwork(sizes={}, probabilities={})".format(self.sizes, self.p)

    def __init__(self, sizes, probabilities, data=None, **attr):
        nx.Graph.__init__(self, data, **attr)

        self.node_groups = {}
        self.se_sets = {}
        self.ce_sets = {}
        self.sizes = sizes
        self.n = len(sizes)
        self.p = np.array(probabilities)
        self.color_annotation = "color"

        prev = 0
        for group, acc in enumerate(accumulate(sizes)):
            self.node_groups[group] = list(range(prev, acc))
            prev = acc

        for i, node_group in self.node_groups.items():
            self.se_sets[i] = []
            for x in combinations(node_group, 2):
                if uniform() < self.p[i, i]:
                    self.se_sets[i].append(x)

        for c1, c2 in combinations(range(self.n), 2):
            self.ce_sets[c1, c2] = []
            for x in product(self.node_groups[c1], self.node_groups[c2]):
                if uniform() < self.p[c1, c2]:
                    self.ce_sets[c1, c2].append(x)

        for i, node_group in self.node_groups.items():
            for node in node_group:
                self.add_node(node, {self.color_annotation: i})

        for se_set in self.se_sets.values():
            for edge in se_set:
                self.add_edge(*edge)

        for ce_set in self.ce_sets.values():
            for edge in ce_set:
                self.add_edge(*edge)

    def draw(self, colors, pos=None, title=None, node_size=300):
        if len(colors) < self.n:
            raise Exception("Wrong number of colors")

        if pos is None:
            pos = nx.spring_layout(self, iterations=100, )

        for i, node_group in self.node_groups.items():
            nx.draw_networkx_nodes(self, pos, nodelist=node_group, node_color=colors[i], node_size=node_size)

        for i, se_set in self.se_sets.items():
            nx.draw_networkx_edges(self, pos, edgelist=se_set, edge_color=colors[i], width=2, alpha=0.3)

        for i, ce_set in self.ce_sets.items():
            nx.draw_networkx_edges(self, pos, edgelist=ce_set, edge_color='grey', width=2, alpha=0.2)

        if title:
            plt.title(title)

        plt.axis('off')
        plt.show()
