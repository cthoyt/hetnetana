import json
import logging
import os
from collections import defaultdict

from py2neo import Node, Relationship

from .multihetnet import MultiHetNet, COLOR, ANNOTATIONS, TYPE

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


def get_resources(resources):
    if isinstance(resources, str):
        with open(os.path.expanduser(resources)) as resource_file:
            return json.load(resource_file)
    elif isinstance(resources, dict):
        return resources
    return json.load(resources)


def load_nodes(hn, fp, color, attributes=None, delimiter='\t'):
    """
    :param hn: a hetnet
    :param fp: a file path
    :param color:
    :param attributes:
    :param delimiter:
    :return:
    """
    attributes = attributes if attributes else {}

    with open(os.path.expanduser(fp)) as f:
        _, *header = next(f).strip().split(delimiter)
        for row in f:
            node, *features = row.strip().split(delimiter)
            annotations = {attribute: features[column_idx] for attribute, column_idx in attributes.items()}
            hn.add_node(node, attr_dict={COLOR: color, ANNOTATIONS: annotations})


def from_resource(resources):
    resources = get_resources(resources)
    bp = os.path.expanduser(os.path.expandvars(resources.get('base_path', '')))
    params = resources['params']
    hn = MultiHetNet(params=params)

    color_annotation_default = defaultdict(dict)
    for color in params.keys():
        for annotation in params[color]:
            if 'default' in params[color][annotation]:
                color_annotation_default[color][annotation] = params[color][annotation]['default']

    for edge_file in resources['edge_files']:
        log.info("Loading edge file: {}".format(edge_file['path']))

        index2color = {}
        for color, indexes in edge_file['indexes'].items():
            for index in indexes:
                index2color[index] = color

        ci_index, cj_index = sorted(index2color)
        ci_color, cj_color = index2color[ci_index], index2color[cj_index]

        ci_defaults = color_annotation_default[ci_color]
        cj_defaults = color_annotation_default[cj_color]

        sep = edge_file.get('delimiter', '\t')
        with open(os.path.join(bp, edge_file['path'])) as f:
            if not edge_file.get('skip_header', False):
                _ = next(f)
            for line in f:
                sline = line.strip().split(sep)
                u, v = sline[ci_index], sline[cj_index]

                if u not in hn:
                    hn.add_node(u, attr_dict={COLOR: ci_color, ANNOTATIONS: ci_defaults.copy()})
                if v not in hn:
                    hn.add_node(v, attr_dict={COLOR: cj_color, ANNOTATIONS: cj_defaults.copy()})

                hn.add_edge(u, v, key=edge_file[TYPE])

    for node_file in resources['node_files']:
        log.info("Loading node file: {}".format(node_file['path']))

        if node_file.get('method', '') == 'mark_inclusive':
            annotation = node_file['annotation']
            color = node_file['color']
            value = node_file['value']
            assert value in hn.params[color][annotation]['options']

            idx = node_file.get('index', 0)
            sep = node_file.get('delimiter', ',')

            with open(os.path.join(bp, node_file['path'])) as f:
                _ = next(f)
                nodes_to_mark = {line.strip().split(sep)[idx] for line in f}

            for node_to_mark in nodes_to_mark:
                if node_to_mark in hn:
                    hn.set_annotation(node_to_mark, annotation, value)
                    log.debug(
                        "Marking {} as {}: {}".format(node_to_mark, value, hn.get_annotation(node_to_mark, annotation)))
                else:
                    log.debug("Not marking {} as b/c isolated".format(node_to_mark, value))

        else:
            attrs = node_file.get('attributes', {})
            color = node_file['color']
            load_nodes(hn, os.path.join(bp, node_file['path']), color=color, attributes=attrs)

    return hn


def to_neo4j(mhn, neo_graph, qnames, context):
    tx = neo_graph.begin()
    node_map = {}
    for node in mhn:
        color = mhn.get_color(node)
        annotations = mhn.get_annotations(node)
        color_qname = qnames[color]

        node_map[node] = Node(color_qname, name=node, **annotations)
        tx.create(node_map[node])

    for u, v, key, data in mhn.edges(data=True, keys=True):
        neo_u = node_map[u]
        neo_v = node_map[v]

        rel = Relationship(neo_u, key, neo_v, context_hetnetana=context, **data)
        tx.create(rel)
    tx.commit()


def from_neo4j(resources, neo_graph, qnames, context):
    resources = get_resources(resources)
    params = resources['params']
    hn = MultiHetNet(params=params)

    for color in params:
        # TODO: unsafe. slice the records instead
        annotation_queries = [', a.{0} as {0}'.format(p) for p in params[color]]

        match_query = "MATCH (a:{}) RETURN a.name as name".format(qnames[color])
        node_query = match_query + ''.join(annotation_queries)

        for record in neo_graph.run(node_query).data():
            name = record.pop('name')
            hn.add_node(name, {COLOR: color, ANNOTATIONS: record})

    for d in resources['edge_files']:
        b = []
        for key, values in d['indexes'].items():
            for value in values:
                b.append(key)
        u, v = sorted(b)

        u_qname = qnames[u]
        v_qname = qnames[v]

        key = d['type']
        edge_query = 'MATCH (u:{})-[r:{}]-(v:{}) where r.context_hetnetana = "{}"RETURN u.name as S, v.name as T'.format(
            u_qname, key, v_qname, context)

        for record in neo_graph.run(edge_query).data():
            source, target = sorted([record['S'], record['T']])
            hn.add_edge(source, target, key=key)

    return hn
