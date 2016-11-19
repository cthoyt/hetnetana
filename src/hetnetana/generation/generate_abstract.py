#!/usr/bin/env python

import logging
from itertools import combinations, product, groupby, count

import networkx as nx
import numpy as np
from numpy.random import uniform

from ..struct.hetnet import HetNet

log = logging.getLogger()

COLOR = 'color'


def generate_abstract(sizes, probabilities, colors=None):
    if colors is not None:
        assert len(sizes) == len(colors)
    else:
        colors = list(range(len(sizes)))

    params = {color: {} for color in colors}
    probabilities = np.array(probabilities)

    hn = HetNet(params=params)

    for i, color, size in zip(count(), colors, sizes):
        for j in range(size):
            hn.add_node("{}:{}".format(color, j), {COLOR: color})

        for u, v in product(range(size), repeat=2):
            if uniform() < probabilities[i, i]:
                hn.add_edge("{}:{}".format(color, u), "{}:{}".format(color, v))

    for (i, c1), (j, c2) in combinations(enumerate(colors), 2):
        for u, v in product(range(sizes[i]), range(sizes[j])):
            if uniform() < probabilities[i, j]:
                hn.add_edge("{}:{}".format(c1, u), "{}:{}".format(c2, v))

    return hn


def draw_abstract(hn, colors, pos=None):
    if pos is None:
        pos = nx.spring_layout(hn, iterations=175, k=(2 / len(hn)))

    color_dict = dict(zip(sorted(hn.colors), sorted(colors)))

    for i, nodes in groupby(sorted(hn, key=hn.get_color), key=hn.get_color):
        nx.draw_networkx_nodes(hn,
                               pos,
                               node_size=30,
                               nodelist=list(nodes),
                               node_color=color_dict[i])

    def ge(edge):
        return sorted(map(hn.get_color, edge))

    for (c1, c2), edges in groupby(sorted(hn.edges(), key=ge), key=ge):
        nx.draw_networkx_edges(hn,
                               pos,
                               edgelist=list(edges),
                               edge_color=(color_dict[c1] if c1 == c2 else 'grey'),
                               alpha=(0.6 if c1 == c2 else 0.4))
