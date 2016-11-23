import json
import logging
import os
import time
from collections import defaultdict, Counter
from datetime import datetime
from functools import lru_cache, wraps
from getpass import getuser

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy

from ..constants import COLOR, ANNOTATIONS, TYPE

log = logging.getLogger('hetnetana')


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        start = time.clock()
        res = f(*args, **kwargs)
        log.info("{} finished in {}s".format(f.__name__, round(time.clock() - start, 3)))
        return res

    return wrap


# TODO: multiple inheritance
class MultiHetNet(nx.MultiGraph):
    def __init__(self, *, data=None, params=None, **attr):
        """
        :rtype: HetNet
        :param params: the parameters
        """
        nx.MultiGraph.__init__(self, data=data, **attr)
        self.params = params
        if params:
            self.annotations = {color: list(sorted(annotations.keys())) for color, annotations in params.items()}

    def get_color(self, n):
        return self.node[n][COLOR]

    def get_annotations(self, n):
        return self.node[n][ANNOTATIONS]

    def get_annotation(self, n, attribute, default=None):
        return self.get_annotations(n).get(attribute, default)

    def set_annotation(self, n, attribute, value):
        self.node[n][ANNOTATIONS][attribute] = value

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
        edge_map = defaultdict(list)
        for u, v, key in self.edges(keys=True):
            color_u, color_v = self.get_color(u), self.get_color(v)
            if color_u < color_v:
                edge_map[color_u, color_v, key].append((u, v))
            else:
                edge_map[color_v, color_u, key].append((v, u))
        return edge_map

    def node_matches(self, n, color, attributes=None):
        if color != self.get_color(n):
            return False
        elif attributes is None:
            return True
        ma = self.get_annotations(n)
        return all(annot in ma and ma[annot] == attr for annot, attr in attributes.items())

    def match_simple_cp_path(self, node, cp):
        if 0 == len(cp):
            yield node,
        else:
            key, color, *head = cp
            for neighbor in self.neighbors(node):
                if self.get_color(neighbor) == color and key in self.edge[node][neighbor]:
                    for path in self.match_simple_cp_path(neighbor, head):
                        if node not in path:
                            yield (node,) + path

    def match_terminal_cp_path(self, node, cp):
        head = cp[:-2]
        edge_key, (tail_color, tail_annotations) = cp[-2:]
        paths = []
        for *path, path_terminal in self.match_simple_cp_path(node, head):
            for neighbor in self.neighbors(path_terminal):
                if edge_key in self.edge[path_terminal][neighbor] and \
                        self.node_matches(neighbor, tail_color, dict(tail_annotations)) and neighbor not in path:
                    paths.append(tuple(path) + (path_terminal, neighbor))

        return tuple(paths)

    @lru_cache(maxsize=None)
    def _get_walks_exhaustive(self, node, length):
        if 0 == length:
            return (node,),

        paths = []
        for neighbor in self.neighbors(node):
            for path in self._get_walks_exhaustive(neighbor, length - 1):
                if node in path:
                    continue
                for edge_idx, data in self.edge[node][neighbor].items():
                    paths.append((node, edge_idx) + path)

        return tuple(paths)

    def _colorize_to_simple(self, path):
        t = [self.get_color(path[0])]
        for i in range(1, len(path), 2):
            t.append(path[i])
            t.append(self.get_color(path[i + 1]))
        return tuple(t)

    def colorize_path(self, path, annotations=None, color_path_type="simple"):
        """
        :param path: a sequence of nodes
        :param annotations: a dictionary from color->list of desired annotations
        :param color_path_type:
        :return: a color path matching this path
        """
        # validate_color_path_type(color_path_type)
        if color_path_type == "simple":
            return self._colorize_to_simple(path)
        elif color_path_type == 'terminal':
            return self._colorize_to_terminal(path, annotations)
        elif color_path_type == 'all':
            raise NotImplemented
            # return self._colorize_to_total(path, annotations)

    # TODO: ftlog clean this up
    def _colorize_to_terminal(self, path, annotations=None):
        t = [self.get_color(path[0])]
        for i in range(1, len(path) - 2, 2):
            t.append(path[i])
            t.append(self.get_color(path[i + 1]))
        tprop, tnode = path[-2], path[-1]
        tcolor = self.get_color(tnode)
        t.append(tprop)
        annotations = annotations if annotations is not None else self.annotations
        atc = annotations.get(tcolor, ())
        tna = tuple((annot, val) for annot, val in self.get_annotations(tnode).items() if annot in atc)
        t.append((tcolor, tna))

        return tuple(t)

    def get_walks(self, node, length, stochastic=False, n_walks=None):
        if stochastic:
            log.debug("Number walks: {}".format(n_walks))
            raise NotImplementedError
        return self._get_walks_exhaustive(node, length)

    @timing
    def calculate_footprints(self, nodes=None, max_length=None, color_path_type=None, annotations=None,
                             n_walks=None, stochastic=False, normalize=False, entropies=False):
        """
        Calculate the topological footprints for the given nodes

        :param nodes: nodes to interrogate
        :param max_length: maximum path length to explore. defaults to O(ln(ln(N)) where N is the size of the network
        :param color_path_type: type of color path to generate
        :param annotations: annotations to use if color path type is 'terminal' or 'total'. defaults to all annotations.
        :param stochastic: use stochastic sampling technique? defaults to false, and an exhaustive search is performed.
                            A stochastic approach might be more appropriate for enormous networks.
        :param n_walks: number of random walks to take if using a stochastic approach
        :param normalize: normalize counts for each path length to percentage observed. Stochastic observations are
                            automatically normalized because their unnormalized interpretation is nonsensical.
        :param entropies: additionally calculate entropies
        :return: a pandas DataFrame containing the features
        """

        if nodes is None:
            nodes = self.nodes()
        elif isinstance(nodes, str):
            nodes = self.get_color_map()[nodes]

        max_length = max_length if max_length is not None else 1 + int(0.9 * np.ceil(np.log(np.log(len(self)))))
        annotations = annotations if annotations is not None else self.annotations
        color_path_type = color_path_type if color_path_type is not None else 'terminal'
        n_walks = n_walks if n_walks is not None else 10000

        self.graph['latest_footprint_config_meta'] = {
            'analysis_start': str(datetime.now()),
            'analyser': getuser(),
            'max_length': max_length,
            'color_path_type': color_path_type,
            'stochastic': stochastic
        }

        log.info("Calculating footprints with parameters: {}".format(
            json.dumps(self.graph['latest_footprint_config_meta'], indent=2)))

        features = {}
        for node in nodes:
            features[node] = {}
            for length in range(1, max_length + 1):
                walks = self.get_walks(node, length, stochastic=stochastic, n_walks=n_walks)

                # Group walks by color path
                color_path_grouping = defaultdict(list)
                for path in walks:
                    color_path = self.colorize_path(path, annotations, color_path_type=color_path_type)
                    color_path_str = encode_color_path(color_path[1:], color_path_type)
                    color_path_grouping[color_path_str].append(color_path)

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

        features_df = pd.DataFrame.from_dict(dict(features)).T.fillna(0).sort_index(axis=0).sort_index(axis=1)

        self.graph['latest_footprint_config_specifics'] = {
            'descriptors': features_df.columns,
            'nodes_analysed': sorted(nodes)
        }

        # log.debug('Nodes analyzed: {}'.format(nodes))

        return features_df

    def to_resource(self, directory, sep='\t'):
        directory = os.path.expanduser(directory) if directory is not None else None
        resources = {
            'notes': self.graph,
            'base_path': directory,
            'node_files': [],
            'edge_files': [],
            'params': self.params
        }

        for color, nodes in self.get_color_map().items():

            fname = "{}.tsv".format(color)
            fpath = os.path.join(directory, fname)
            data = {'path': fpath, 'color': color}

            attributes = self.annotations[color]
            if attributes:
                data['attributes'] = {attr: i for i, attr in enumerate(attributes)}
            resources['node_files'].append(data)

            with open(fpath, 'w') as f:
                print(color, *attributes, file=f, sep=sep)
                for node in sorted(nodes):
                    print(node, *[self.get_annotation(node, a) for a in attributes], file=f, sep=sep)

        for (ca, cb, key), edges in self.get_edge_map().items():
            fname = '{}_{}_{}.tsv'.format(ca, key, cb)
            fpath = os.path.join(directory, fname)
            data = {
                'path': fpath,
                TYPE: key,
                'indexes': defaultdict(list)
            }

            data['indexes'][ca].append(0)
            data['indexes'][cb].append(1)

            resources['edge_files'].append(data)
            with open(fpath, 'w') as f:
                print(ca, cb, file=f, sep=sep)
                for a, b in edges:
                    print(a, b, file=f, sep=sep)

        resource_path = os.path.join(directory, 'resources.json')
        with open(resource_path, 'w') as f:
            json.dump(resources, f, indent=4)

        return resource_path


def pairwise(iterable):
    it = iter(iterable)
    for a, b in zip(it, it):
        yield a, b


def encode_color_path(color_path, color_path_type="simple", entry_sep="&", sep="|", attr_sep=";", kv_sep=":",
                      property_sep='_'):
    # validate_color_path_type(color_path_type)

    if 0 == len(color_path):
        return ''
    elif color_path_type == 'simple':
        return entry_sep.join('{}{}{}'.format(a, property_sep, b) for a, b in pairwise(color_path))
    elif color_path_type == "terminal":
        cp_head = color_path[:-2]
        terminal_property, (terminal_color, terminal_classes) = color_path[-2:]
        if cp_head:
            s = encode_color_path(cp_head, color_path_type='simple', entry_sep=entry_sep, sep=sep, attr_sep=attr_sep,
                                  kv_sep=kv_sep, property_sep=property_sep)
            s += entry_sep
        else:
            s = ''
        s += str(terminal_property)
        s += property_sep
        s += str(terminal_color)
        if terminal_classes:
            s += sep
            s += attr_sep.join("{}{}{}".format(k, kv_sep, v) for k, v in terminal_classes)
        return s
    elif color_path_type == "all":
        return entry_sep.join("{}{}{}".format(
            color, sep, attr_sep.join("{}{}{}".format(a, kv_sep, b)
                                      for a, b in sorted(annotations)))
                              for color, annotations in color_path)
