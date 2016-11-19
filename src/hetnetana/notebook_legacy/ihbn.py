from itertools import combinations, product, accumulate

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import uniform


class IHBNetwork(nx.Graph):
    def __repr__(self):
        return "IHBNetwork(sizes={}, probabilities={})".format(self.sizes, self.p)

    def __init__(self, sizes, p, iflags={}, data=None, **attr):
        nx.Graph.__init__(self, data, **attr)
        self.color_annotation = "color"

        assert len(sizes) == len(p)
        self.p = np.array(p)
        self.iflags = iflags

        self.node_groups = {}
        self.sizes = sizes
        self.se_sets = {}
        self.ce_sets = {}
        self.internodes = {}
        self.interedges = {}
        self.n = len(sizes)
        self.icolors = {}

        # build regular nodes
        prev = 0
        for group, acc in enumerate(accumulate(sizes)):
            self.node_groups[group] = list(range(prev, acc))
            prev = acc

        # build internodes
        self.q = sum(sizes)
        icolorc = 0
        for i, j in iflags:
            self.internodes[i, j] = []
            self.interedges[i, j] = []
            icolorc -= 1
            self.icolors[i, j] = icolorc
            for a, b in product(self.node_groups[i], self.node_groups[j]):
                if uniform() < self.p[i, j]:
                    self.internodes[i, j].append(self.q)
                    self.interedges[i, j].append((self.q, a))
                    self.interedges[i, j].append((self.q, b))

                    self.add_node(self.q, {self.color_annotation: icolorc})
                    self.add_edge(self.q, a)
                    self.add_edge(self.q, b)

                    self.q += 1

        # build edges
        for i, node_group in self.node_groups.items():
            if (i, i) in self.iflags:
                continue
            self.se_sets[i] = []
            for x in combinations(node_group, 2):
                if uniform() < self.p[i, i]:
                    self.se_sets[i].append(x)

        for c1, c2 in combinations(range(self.n), 2):
            if (c1, c2) in self.iflags or (c2, c1) in self.iflags:
                continue
            self.ce_sets[c1, c2] = []
            for x in product(self.node_groups[c1], self.node_groups[c2]):
                if uniform() < self.p[c1, c2]:
                    self.ce_sets[c1, c2].append(x)

        # populate networkx graph
        for i, node_group in self.node_groups.items():
            for node in node_group:
                self.add_node(node, {self.color_annotation: i})

        for se_set in self.se_sets.values():
            for edge in se_set:
                self.add_edge(*edge)

        for ce_set in self.ce_sets.values():
            for edge in ce_set:
                self.add_edge(*edge)

    def rnodes(self):
        """ Returns real nodes (no internodes)"""
        return [n for n in self.nodes() if self.node[n][self.color_annotation] >= 0]

    def rneighbors(self, s):
        """ Returns real neighbors (no internodes)"""
        a = set()
        for u in self.neighbors(s):
            if 0 <= self.node[u][self.color_annotation]:
                a.add(u)
            else:
                for v in self.neighbors(u):
                    if v != s:
                        a.add(v)
        return list(a)

    def draw(self, colors, pos=None, title=None, **attr):
        if len(colors) < self.n:
            raise Exception("Wrong number of colors")

        if pos is None:
            pos = nx.spring_layout(self)

        for i, node_group in self.node_groups.items():
            nx.draw_networkx_nodes(self, pos, nodelist=node_group, node_color=colors[i])

        for i, se_set in self.se_sets.items():
            nx.draw_networkx_edges(self, pos, edgelist=se_set, edge_color=colors[i], width=3, alpha=0.5)

        for i, ce_set in self.ce_sets.items():
            nx.draw_networkx_edges(self, pos, edgelist=ce_set, edge_color='grey', width=3, alpha=0.5)

        for i, j in self.iflags:
            nx.draw_networkx_nodes(self, pos, nodelist=self.internodes[i, j], node_color='black', node_size=15)
            nx.draw_networkx_edges(self, pos, edgelist=self.interedges[i, j], edge_color='black', alpha=0.5)

        if title:
            plt.title(title)

        plt.axis('off')
        plt.show()
