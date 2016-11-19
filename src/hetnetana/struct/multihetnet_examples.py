from hetnetana import hgnc, mi
from .multihetnet import MultiHetNet

DOWN, UP, UNREGULATED = 'DOWN', 'UP', 'UNREGULATED'


def quick_adders(n):
    def g(i, **kwargs):
        n.add_node(hgnc(i), dict(color=hgnc.color, annotations=kwargs))

    def m(i, **kwargs):
        n.add_node(mi(i), dict(color=mi.color, annotations=kwargs))

    def e(i, j, key=None, attr_dict=None, **kwargs):
        n.add_edge(i, j, key, attr_dict, **kwargs)

    return g, m, e


edge_color_map = {
    'ppi': 'purple',
    'coexp': 'blue',
    'encode': 'green',
    'mti': 'red'
}


def color_edges(multigraph, key_color_mapping):
    for u, v, key in multigraph.edges(keys=True):
        multigraph.edge[u][v][key]['color'] = key_color_mapping[key]


def generate_example_1():
    mhn = MultiHetNet(params={
        hgnc.color: {
            'regulation': {'options': [DOWN, UP, UNREGULATED]}
        },
        mi.color: {
            'c1': {'options': ['a', 'b', 'c']},
            'c2': {'options': ['d', 'e', 'f']}
        }
    })

    g, m, e = quick_adders(mhn)

    g(1, regulation=UP)
    g(2, regulation=UNREGULATED)
    g(3, regulation=DOWN)
    g(4, regulation=UNREGULATED)
    g(5, regulation=UNREGULATED)
    g(6, regulation=UNREGULATED)
    g(7, regulation=UP)
    g(8, regulation=UNREGULATED)
    g(9, regulation=UNREGULATED)
    g(10, regulation=UNREGULATED)
    g(11, regulation=UNREGULATED)
    g(12, regulation=UNREGULATED)
    g(13, regulation=UNREGULATED)
    g(14, regulation=UP)
    g(15, regulation=UNREGULATED)
    g(16, regulation=UNREGULATED)
    g(17, regulation=UNREGULATED)
    g(18, regulation=DOWN)
    g(19, regulation=UNREGULATED)
    g(20, regulation=UNREGULATED)

    m(17, c1='a', c2='f')
    m(18, c1='a', c2='e')
    m(19, c1='n', c2='d')
    m(20, c1='n', c2='e')

    e(mi(17), hgnc(17), key='encode')
    e(mi(18), hgnc(18), key='encode')
    e(mi(19), hgnc(19), key='encode')
    e(mi(20), hgnc(20), key='encode')

    e(hgnc(3), mi(17), key='mti')
    e(hgnc(9), mi(17), key='mti')
    e(hgnc(4), mi(17), key='mti')
    e(hgnc(7), mi(18), key='mti')
    e(hgnc(16), mi(19), key='mti')
    e(hgnc(7), mi(19), key='mti')
    e(hgnc(5), mi(20), key='mti')
    e(hgnc(13), mi(20), key='mti')

    e(hgnc(1), hgnc(2), key='ppi')
    e(hgnc(2), hgnc(6), key='ppi')
    e(hgnc(5), hgnc(6), key='ppi')
    e(hgnc(6), hgnc(7), key='ppi')
    e(hgnc(7), hgnc(12), key='ppi')
    e(hgnc(6), hgnc(12), key='ppi')
    e(hgnc(12), hgnc(13), key='ppi')
    e(hgnc(12), hgnc(14), key='ppi')
    e(hgnc(14), hgnc(15), key='ppi')
    e(hgnc(7), hgnc(10), key='ppi')
    e(hgnc(10), hgnc(11), key='ppi')
    e(hgnc(11), hgnc(16), key='ppi')
    e(hgnc(8), hgnc(10), key='ppi')
    e(hgnc(8), hgnc(9), key='ppi')
    e(hgnc(9), hgnc(10), key='ppi')
    e(hgnc(3), hgnc(8), key='ppi')
    e(hgnc(3), hgnc(9), key='ppi')
    e(hgnc(3), hgnc(4), key='ppi')

    e(hgnc(5), hgnc(13), key='coexp')
    e(hgnc(1), hgnc(2), key='coexp')
    e(hgnc(5), hgnc(6), key='coexp')
    e(hgnc(3), hgnc(4), key='coexp')
    e(hgnc(4), hgnc(8), key='coexp')
    e(hgnc(8), hgnc(9), key='coexp')
    e(hgnc(8), hgnc(10), key='coexp')
    e(hgnc(10), hgnc(16), key='coexp')
    e(hgnc(7), hgnc(10), key='coexp')
    e(hgnc(15), hgnc(19), key='coexp')

    color_edges(mhn, edge_color_map)

    return mhn


def generate_example_2():
    mhn = MultiHetNet(params={
        hgnc.color: {},
        mi.color: {}
    })

    mhn.add_node(hgnc(1), dict(color=hgnc.color))
    mhn.add_node(hgnc(2), dict(color=hgnc.color))
    mhn.add_node(mi(1), dict(color=mi.color))
    mhn.add_node(mi(2), dict(color=mi.color))
    mhn.add_node(mi(3), dict(color=mi.color))

    mhn.add_edge(hgnc(1), hgnc(2), key='ppi')
    mhn.add_edge(hgnc(1), hgnc(2), key='coexp')
    mhn.add_edge(hgnc(1), mi(1), key='mti')
    mhn.add_edge(hgnc(1), mi(2), key='mti')
    mhn.add_edge(hgnc(2), mi(3), key='mti')

    return mhn


def generate_example_4():
    mhn = MultiHetNet(params={
        hgnc.color: {},
        mi.color: {}
    })

    mhn.add_node(hgnc(1), dict(color=hgnc.color))
    mhn.add_node(hgnc(2), dict(color=hgnc.color))
    mhn.add_node(hgnc(3), dict(color=hgnc.color))
    mhn.add_node(hgnc(4), dict(color=hgnc.color))
    mhn.add_node(hgnc(5), dict(color=hgnc.color))

    mhn.add_node(mi(1), dict(color=mi.color))

    mhn.add_edge(hgnc(2), hgnc(4), key='coexp')

    mhn.add_edge(hgnc(1), hgnc(2), key='ppi')
    mhn.add_edge(hgnc(1), hgnc(3), key='ppi')
    mhn.add_edge(hgnc(2), hgnc(3), key='ppi')
    mhn.add_edge(hgnc(3), hgnc(4), key='ppi')
    mhn.add_edge(hgnc(5), mi(1), key='encode')
    mhn.add_edge(mi(1), hgnc(2), key='mti')
    mhn.add_edge(mi(1), hgnc(3), ket='mti')

    return mhn
