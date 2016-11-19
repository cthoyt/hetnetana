#!/usr/bin/env python

"""Heterogeneous network data wrapper"""

import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from getpass import getuser

import networkx as nx
import numpy as np
import pandas as pd
from numpy import random
from scipy.stats import entropy

log = logging.getLogger()

ANNOTATIONS = 'annotations'
COLOR = 'color'
TYPE = 'type'


class HetNet(nx.Graph):
    """Wrapper of networkx graph for dealing with heterogeneous networks"""

    def __init__(self, params=None):
        """
        :rtype: HetNet
        :param params: the parameters
        """
        super().__init__(self)
        self.params = params
        self.colors = set(self.params.keys())
        self.annotations = {color: list(sorted(annots.keys())) for color, annots in self.params.items()}

    def __repr__(self):
        return "HetNet(nodes={}, edges={}, colors={})".format(len(self), self.number_of_edges(), sorted(self.colors))

    """ DATA ACCESSORS """

    def get_color(self, node):
        return self.node[node][COLOR]

    def get_annotations(self, n):
        return self.node[n].get(ANNOTATIONS, {})

    def get_annotation(self, n, attribute, default=None):
        return self.get_annotations(n).get(attribute, default)

    def node_matches(self, node, color, attributes=None):
        if color == self.get_color(node):
            if attributes is None:
                return True
            ma = self.get_annotations(node)
            return all(annot in ma and ma[annot] == attr for annot, attr in attributes.items())
        return False

    def get_nodes(self, color, attributes=None):
        """
        :param color:
        :param attributes:
        :return: nodes with matching color and attributes
        """
        return [node for node in self if self.node_matches(node, color, attributes)]

    def get_color_map(self):
        """
        :return: a map from color to list of nodes of that color
        """
        color_map = defaultdict(list)
        for n in self:
            color = self.get_color(n)
            color_map[color].append(n)
        return color_map

    def get_edge_map(self):
        """
        :return: a map from color pair to list of edges matching that
        """
        edge_map = defaultdict(list)
        for a, b in self.edges():
            ca, cb = self.get_color(a), self.get_color(b)
            if ca < cb:
                edge_map[ca, cb].append((a, b))
            else:
                edge_map[cb, ca].append((b, a))
        return edge_map

    """ IO FUNCTIONS """

    @staticmethod
    def from_resource(resource_path):
        with open(os.path.expanduser(resource_path)) as resource_file:
            resources = json.load(resource_file)
            bp = os.path.expanduser(os.path.expandvars(resources.get('base_path', '')))
            hnn = HetNet(resources['params'])

            for node_file in resources['node_files']:
                attrs = node_file.get('attributes', None)
                color = node_file['color']
                hnn.load_nodes(os.path.join(bp, node_file['path']), color, attrs)

            for edge_path in resources['edge_files']:
                hnn.load_edges(os.path.join(bp, edge_path['path']))

            return hnn

    def to_resource(self, base_dir=None, sep="\t"):
        log.info("dumping HetNet to {}".format(base_dir))
        base_dir = os.path.expanduser(base_dir) if base_dir is not None else None
        resources = {
            'params': self.params,
            'node_files': [],
            'edge_files': [],
            'base_path': base_dir,
            'notes': self.graph
        }

        for color, nodes in self.get_color_map().items():
            attributes = sorted(self.params[color].keys())
            file_name = "{}.tsv".format(color)
            file_path = os.path.join(base_dir, file_name)
            data = {'path': file_name, 'color': color}
            if attributes:
                data['attributes'] = attributes
            resources['node_files'].append(data)

            with open(file_path, 'w') as f:
                print(color, *attributes, file=f, sep=sep)
                for node in sorted(nodes):
                    print(node, *[self.get_annotation(node, a) for a in attributes], file=f, sep=sep)

        for (ca, cb), edges in self.get_edge_map().items():
            file_name = "{}_{}.tsv".format(ca, cb)
            file_path = os.path.join(base_dir, file_name)
            resources['edge_files'].append({'path': file_name})
            with open(file_path, 'w') as f:
                print(ca, cb, file=f, sep=sep)
                for a, b in sorted(edges):
                    print(a, b, file=f, sep=sep)

        with open(os.path.join(base_dir, "resources.json"), 'w') as f:
            json.dump(resources, f, indent=4)

    def to_lists(self, base_dir, sep='\t'):
        base_dir = os.path.expanduser(base_dir)
        with open(os.path.join(base_dir, 'edgelist.tsv'), 'w') as f:
            print('source', 'target', file=f, sep=sep)
            for a, b in self.edges():
                print(a, b, file=f, sep=sep)

        with open(os.path.join(base_dir, 'nodelist.tsv'), 'w') as f:
            print('node', 'class', file=f, sep=sep)
            for node, data in self.nodes(data=True):
                print(node, data['color'], file=f, sep=sep)

    def load_nodes(self, file, color, attributes=None, delimiter="\t"):
        """
        Load nodes from a node-list file. Note: each node list file must contain only one color node.
        :param file: the path to a node file
        :param color: the color of the nodes in this file
        :param attributes: a dictionary of attributes' columns to keep. Defaults to keeping none.
        :param delimiter: the delimiter
        """
        attributes = attributes if attributes else []

        with open(os.path.expanduser(file)) as f:
            _, *header = next(f).strip().split(delimiter)
            for row in f:
                node, *features = row.strip().split(delimiter)
                annotations = {attr: value for attr, value in zip(header, features) if attr in attributes}
                self.add_node(node, {"color": color, "annotations": annotations})

    def load_edges(self, file, delimiter="\t"):
        """
        Loads edges from a two-column csv
        :param file the file containing two columns for the nodes then additional attributes
        :param delimiter:
        """
        with open(os.path.expanduser(file)) as f:
            ha, hb, *attributes = next(f).strip().split(delimiter)
            for line in f:
                i, j, *features = line.strip().split(delimiter)
                self.add_edge(i, j, dict(zip(attributes, features)))

    """ NETWORK METHODS """

    def multi_match_color_paths(self, color_path, color_path_type="simple"):
        """
        Exhaustively final all paths for every node maching a given color path
        :param color_path: color path
        :param color_path_type: color path type
        :return: dictionary of node->iterable of paths
        """
        return {node: self.match_color_paths(node, color_path, color_path_type=color_path_type) for node in self}

    def match_color_paths(self, node, color_path, color_path_type='simple'):
        """
        Exhaustively find all paths starting at a given node that match a given color path
        :param node:
        :param color_path:
        :param color_path_type:
        :return:
        """
        assert node in self, '{} not in network'.format(node)
        validate_color_path(color_path, color_path_type, valid_colors=self.colors)

        if color_path_type == 'simple':
            return self._match_simple_color_paths(node, color_path)
        elif color_path_type == 'terminal':
            return self._match_terminal_color_paths(node, color_path)
        elif color_path_type == 'all':
            return self._match_total_color_paths(node, color_path)

    def _match_simple_color_paths(self, node, color_path):
        """
        match simple color paths (containing no annotations)
        :param node: start node
        :param color_path: simple color path
        :return: all paths matching w
        """
        if 0 == len(color_path):
            return (node,),
        paths = []
        for neighbor in self.neighbors(node):
            if self.get_color(neighbor) == color_path[0]:
                for path in self._match_simple_color_paths(neighbor, color_path[1:]):
                    paths.append((node,) + path)
        return tuple(paths)

    def _match_terminal_color_paths(self, node, terminal_color_path):
        """
        :param node:
        :param terminal_color_path:
        :return: all paths originating at node matching complex color path
        """
        terminal_color, terminal_annotations = terminal_color_path[-1]
        terminal_annotations = dict(terminal_annotations)
        paths = []
        for path in self._match_simple_color_paths(node, terminal_color_path[:-1]):
            for neighbor in self.neighbors(path[-1]):
                if self.node_matches(neighbor, terminal_color, terminal_annotations):
                    paths.append(path + (neighbor,))
        return tuple(paths)

    def _match_total_color_paths(self, node, total_color_path):
        if 0 == len(total_color_path):
            return (node,),
        paths = []
        color, attributes = total_color_path[0]
        attributes = dict(attributes)
        for neighbor in self.neighbors(node):
            if self.node_matches(neighbor, color, attributes):
                for path in self._match_total_color_paths(neighbor, total_color_path[1:]):
                    paths.append((node,) + path)
        return tuple(paths)

    def random_walk(self, node, length):
        """
        Produces a random walk that won't include duplicate nodes.
        Assumes node is in a connected component with size of at least 'length'
        :param node: starting node
        :param length: length of random walk
        :return: list of nodes corresponding to walk of given length
        """
        path = [node]
        s = node
        for _ in range(length):
            neighbors = set(self.neighbors(s)).difference(path)
            if 0 == len(neighbors):
                return None
            s = list(neighbors)[random.choice(len(neighbors))]
            path.append(s)
        return tuple(path)

    """ ANALYTICAL METHODS """

    def _colorize_to_simple(self, path):
        return tuple(self.get_color(node) for node in path)

    def _colorize_to_terminal(self, path, annotations=None):
        annotations = annotations if annotations is not None else self.annotations
        head = tuple(self.get_color(node) for node in path[:-1])
        tail_node = path[-1]
        tail_node_color = self.get_color(tail_node)
        atc = annotations.get(tail_node_color, ())
        tail_node_annotations = tuple((annot, val) for annot, val in self.get_annotations(tail_node).items() if
                                      annot in atc)
        return head + ((tail_node_color, tail_node_annotations),)

    def _colorize_to_total(self, path, annotations=None):
        annotations = annotations if annotations is not None else self.annotations
        result = []
        for node in path:
            node_color = self.get_color(node)
            atc = annotations.get(node_color, ())
            node_annotations = self.get_annotations(node)
            a = sorted((attribute, node_annotations[attribute]) for attribute in atc)
            result.append((node_color, tuple(a)))
        return tuple(result)

    def colorize_path(self, path, annotations=None, color_path_type="simple"):
        """
        :param path: a sequence of nodes
        :param annotations: a dictionary from color->list of desired annotations
        :param color_path_type:
        :return: a color path matching this path
        """
        validate_color_path_type(color_path_type)
        if color_path_type == "simple":
            return self._colorize_to_simple(path)
        elif color_path_type == 'terminal':
            return self._colorize_to_terminal(path, annotations)
        elif color_path_type == 'all':
            return self._colorize_to_total(path, annotations)

    def _get_walks_exhaustive(self, node, length):
        """
        Performs an exhaustive search of all paths given length with DFS algorithm
        """
        if 0 == length:
            yield node,
        else:
            for neighbor in self.neighbors(node):
                for path in self._get_walks_exhaustive(neighbor, length - 1):
                    if node not in path:
                        yield (node,) + path

    def _get_walks_stochastic(self, node, length, n_walks=10000):
        """
        Performs a stochastic sampling of all paths of given length with random walks
        """
        assert length <= len(nx.node_connected_component(self, node)), 'component too small for random walks this long'
        for _ in range(n_walks):
            w = self.random_walk(node, length)
            if w is not None:
                yield w

    def get_walks(self, node, length, stochastic=False, n_walks=10000):
        if stochastic:
            return self._get_walks_stochastic(node, length, n_walks=n_walks)
        else:
            return self._get_walks_exhaustive(node, length)

    # TODO: move to analytical class
    def calculate_footprints(self, nodes=None, max_length=None, color_path_type=None, annotations=None,
                             n_walks=10000, stochastic=False, normalize=False, entropies=False):
        """
        Calculate the topological footprints for the given nodes

        :param nodes: nodes to interrogate
        :type nodes: list
        :param max_length: maximum path length to explore. defaults to O(ln(ln(N)) where N is the size of the network
        :type max_length: int
        :param color_path_type: type of color path to generate
        :type color_path_type: str
        :param annotations: annotations to use if color path type is 'terminal' or 'total'. defaults to all annotations.
        :type annotations: dict
        :param stochastic: use stochastic sampling technique? defaults to false, and an exhaustive search is performed.
                            A stochastic approach might be more appropriate for enormous networks.
        :type stochastic: bool
        :param n_walks: number of random walks to take if using a stochastic approach
        :type n_walks: int
        :param normalize: normalize counts for each path length to percentage observed. Stochastic observations are
                          automatically normalized because their unnormalized interpretation is nonsensical.
        :type normalize: bool
        :param entropies: additionally calculate entropies
        :type entropies: boolean
        :return: DataFrame containing the calculated topological features
        """

        if nodes is None:
            nodes = self.nodes()
        elif isinstance(nodes, str):
            nodes = self.get_color_map()[nodes]

        max_length = max_length if max_length is not None else 1 + int(1.25 * np.ceil(np.log(np.log(len(self)))))
        color_path_type = color_path_type if color_path_type is not None else 'terminal'
        n_walks = n_walks if n_walks is not None else 10000

        features = {}
        for node in nodes:
            features[node] = {}
            for length in range(1, max_length + 1):
                walks = self.get_walks(node, length, stochastic=stochastic, n_walks=n_walks)

                # Group walks by color path
                color_path_grouping = defaultdict(list)
                for path in walks:
                    adjusted_path = path[1:]  # remove original node
                    color_path = self.colorize_path(adjusted_path, annotations, color_path_type=color_path_type)
                    color_path_str = encode_color_path(color_path, color_path_type)
                    color_path_grouping[color_path_str].append(adjusted_path)

                # Count number matching paths, and entropy of distribution of unique terminals
                for color_path, matching_paths in color_path_grouping.items():
                    matching_path_unique_terminals = Counter(path[-1] for path in matching_paths)
                    top_count = len(matching_paths)

                    if stochastic or normalize:
                        top_count /= sum(matching_path_unique_terminals.values())

                    if entropies:
                        top_entropy = entropy(list(matching_path_unique_terminals.values()))
                        features[node]["{}_count".format(color_path)] = top_count
                        features[node]["{}_entropy".format(color_path)] = top_entropy
                    else:
                        features[node][color_path] = top_count

        features_df = pd.DataFrame.from_dict(features).T.fillna(0).sort_index(axis=0).sort_index(axis=1)

        self.graph['latest_footprint_config'] = {
            'analysis_time': str(datetime.now()),
            'analyser': getuser(),
            'nodes_analysed': sorted(nodes),
            'max_length': max_length,
            'color_path_type': color_path_type,
            'descriptors': features_df.columns,
            'stochastic': stochastic
        }

        return features_df


def validate_color_path_type(color_path_type):
    if color_path_type not in {'simple', 'terminal', 'all'}:
        raise ValueError('invalid color_path_type: {}'.format(color_path_type))


def validate_color_path(color_path, color_path_type, valid_colors):
    validate_color_path_type(color_path_type)

    valid_color_set = set(valid_colors)

    if color_path_type == 'simple':
        if not all(color in valid_color_set for color in color_path):
            raise ValueError('invalid {} color path: {}'.format(color_path_type, color_path))

    elif color_path_type == 'terminal':
        if not all(color in valid_color_set for color in color_path[:-1]) or not color_path[-1][0] in valid_color_set:
            raise ValueError('invalid {} color path: {}'.format(color_path_type, color_path))

    elif color_path_type == 'all':
        if not all(color[0] in valid_color_set for color in color_path):
            raise ValueError('invalid {} color path: {}'.format(color_path_type, color_path))


# TODO make static method of HetNet class
def encode_color_path(color_path, color_path_type="simple", entry_sep="&", sep="|", attr_sep=";", kv_sep=":"):
    validate_color_path_type(color_path_type)

    if 0 == len(color_path):
        return ""

    if color_path_type == "terminal":
        s = entry_sep.join(map(str, color_path[:-1]))
        terminal_color, terminal_classes = color_path[-1]
        s += entry_sep
        s += str(terminal_color)
        s += sep
        s += attr_sep.join("{}{}{}".format(k, kv_sep, v) for k, v in terminal_classes)
        return s
    elif color_path_type == "all":
        return entry_sep.join("{}{}{}".format(
            color, sep, attr_sep.join("{}{}{}".format(a, kv_sep, b)
                                      for a, b in sorted(annotations)))
                              for color, annotations in color_path)
    elif color_path_type == 'simple':
        return entry_sep.join(color_path)


def decode_color_path(color_path, color_path_type="simple", entry_sep="&", sep="|", attr_sep=";", kv_sep=":"):
    validate_color_path_type(color_path_type)

    color_path = [a for a in color_path.strip().split(entry_sep) if a]

    if color_path_type == 'simple':
        return tuple(color_path)

    elif color_path_type == "terminal":
        head = color_path[:-1]
        tail_color, tail_annotations = color_path[-1].split(sep)
        if not tail_annotations:
            return tuple(head) + tuple([(tail_color, ())])
        tail_annotations = tail_annotations.split(attr_sep)
        tail_annotations = tuple(tuple(x.split(kv_sep)) for x in tail_annotations if x)
        return tuple(head) + tuple([(tail_color, tail_annotations)])

    elif color_path_type == "all":
        color_path = [e.split(sep) for e in color_path if e]
        color_path = [(a, tuple(tuple(d for d in c.split(kv_sep) if d) for c in b.split(attr_sep) if c)) for a, b in
                      color_path]
        return tuple(color_path)
