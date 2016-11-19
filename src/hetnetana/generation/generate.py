#!/usr/bin/env python

""" Generate a network with specific color features """

import itertools as itt
import logging
from collections import Counter

import numpy as np

from ..struct.hetnet import encode_color_path, validate_color_path

log = logging.getLogger()

""" STATISTICAL """


def get_distribution(graph, color_path, color_path_type="simple"):
    """
    :param graph: HetNet
    :param color_path: color path
    :param color_path_type: color path type
    :return: histogram of number of paths matching color_path
    """
    paths = graph.multi_match_color_paths(color_path, color_path_type)
    return Counter(len(v) for k, v in paths.items())


def get_percentile(graph, color_path, percentile, color_path_type="simple"):
    d = get_distribution(graph, color_path, color_path_type)
    return np.percentile(list(d.elements()), percentile, interpolation='nearest')


def get_percentile_diff(graph, color_path, upper_percentile, lower_percentile, color_path_type="simple"):
    upper_value, lower_value = get_percentile(graph, color_path, [upper_percentile, lower_percentile], color_path_type)
    return upper_value, lower_value, np.abs((upper_value - lower_value) / (upper_percentile - lower_percentile))


""" FEATURE INDUCTION """


def get_inducible_pairs(graph, node, color_path, n_edges, color_path_type='simple'):
    """
    Gets inducible terminal pairs then selects n_edges pairs from their product

    :param graph: graph
    :param node: starting node
    :param color_path: color path (terminal)
    :param n_edges: number edges
    :param color_path_type
    :return:
    """
    pairs = list(get_inducible_pairs_total(graph, node, color_path, color_path_type))

    log.debug('node {} inducing {} has {} induction edges'.format(node,
                                                                  encode_color_path(color_path, color_path_type),
                                                                  len(pairs)))

    if 0 == len(pairs):
        log.warn('no possible paths to induce')
        return []
    elif n_edges > len(pairs):
        log.warn('only {} possible paths to induce. proceeding, but will be saturated and probably not useful'.format(
            len(pairs)))
        return pairs
    else:
        ind = np.random.choice(np.arange(len(pairs)), size=n_edges, replace=False)
        return [pairs[i] for i in ind]


# TODO do this in reverse order and memoize results for matching color paths
def get_inducible_pairs_total(graph, node, color_path, color_path_type='simple'):
    l = len(color_path)
    for i in range(l):
        yield from itt.product(*get_inducible_pairs_indexed(graph, node, color_path, color_path_type, split=i))


def get_inducible_pairs_indexed(graph, node, color_path, color_path_type='simple', split=-1):
    """
    Go further than just matching color_path[:-1] terminals to color_path[-1]

    for i, e in enumerate(color_path)
        match color_path[:i] terminals to nodes with paths matching color_path[i:]

    :param graph:
    :param node: starting node
    :param color_path: the color path to search
    :param color_path_type: type of the color path to search
    :param split: place to split color path and find inducible edges.
                can be set to negative -> gets modded by len(color_path)
    :return:
    """
    validate_color_path(color_path, color_path_type, graph.colors)

    assert -len(color_path) <= split < len(color_path), 'invalid split argument'
    split %= len(color_path)

    if color_path_type in ('simple', 'all'):
        head = color_path[:split]
        head_terminals = []
        for path in graph.match_color_paths(node, head, color_path_type):
            head_terminals.append(path[-1])

        tail = color_path[split:]
        tail_starts = []

        if 1 == len(tail):
            tail_starts = graph.get_nodes(tail[0])
        else:
            for tn in graph.get_nodes(tail[0]):
                if 0 < len(graph.match_color_paths(tn, tail[1:], color_path_type)):
                    tail_starts.append(tn)

        return set(head_terminals), set(tail_starts)
    elif color_path_type == 'terminal':
        if split == len(color_path) - 1:
            paths = graph.match_color_paths(node, color_path[:-1], color_path_type='simple')
            head_terminals = [path[-1] for path in paths]
            induce_path_end_color, induce_path_end_annotations = color_path[-1]
            tail_starts = graph.get_nodes(induce_path_end_color, dict(induce_path_end_annotations))
            return set(head_terminals), set(tail_starts)
        else:
            head = color_path[:split]
            head_terminals = []
            for path in graph.match_color_paths(node, head, color_path_type='simple'):
                head_terminals.append(path[-1])

            # Part B:

            tail = color_path[split:]
            tail_starts = []
            for tn in graph.get_nodes(tail[0]):
                paths = graph.match_color_paths(tn, tail[1:], color_path_type='terminal')
                if 0 < len(paths):
                    tail_starts.append(tn)

            return set(head_terminals), set(tail_starts)
