from .hetnet import HetNet
from .multihetnet import MultiHetNet


def convert(simple_hetnet, collapse_color, collapse_source, collapse_keys, edge_annotations):
    """
    Collapses a simple heterogeneous network along
    - assumes one to one mapping of source to color

    :param collapse_keys:
    :param collapse_color:
    :param collapse_source:
    :param edge_annotations
    :param simple_hetnet: simple heterogeneous network
    :type simple_hetnet: HetNet
    :return: collapsed network
    :rtype: MultiHetNet
    """

    params = simple_hetnet.params.copy()
    del params[collapse_color]
    multi_hetnet = MultiHetNet(params=params)

    node_map = simple_hetnet.get_color_map()
    del node_map[collapse_color]

    for color, nodes in node_map.items():
        for node in nodes:
            multi_hetnet.add_node(node, {
                'color': color,
                'annotations': simple_hetnet.get_annotations(node)
            })

    edge_map = simple_hetnet.get_edge_map()

    if (collapse_source, collapse_color) in edge_map:
        encoding = {b: a for a, b in edge_map.pop((collapse_source, collapse_color))}
    else:
        encoding = dict(edge_map.pop((collapse_color, collapse_source)))

    if collapse_color in collapse_keys:
        self_key = collapse_keys.pop(collapse_color)
        for a, b in edge_map.pop((collapse_color, collapse_color)):
            multi_hetnet.add_edge(encoding[a], encoding[b], key=self_key)

    for color_key in collapse_keys:
        edge_key = collapse_keys[color_key]
        if (collapse_color, color_key) in edge_map:
            edges = edge_map.pop((collapse_color, color_key))
        else:
            edges = ((b, a) for a, b in edge_map.pop((color_key, collapse_color)))
        for a, b in edges:
            multi_hetnet.add_edge(encoding[a], b, key=edge_key)

    for a, b in edge_annotations.items():
        for c, d in b.items():
            if (a, c) in edge_map:
                x = edge_map[a, c]
            else:
                x = [(r, s) for s, r in edge_map[c, a]]

            for x, y in x:
                multi_hetnet.add_edge(x, y, key=d)

    # TODO: automatic handling of remaining edges without user-specified keys

    return multi_hetnet
