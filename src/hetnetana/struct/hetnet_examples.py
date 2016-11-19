from itertools import chain

from hetnetana import *

DOWN, UP, UNREGULATED = "DOWN", "UP", "UNREGULATED",


def quick_adders(n):
    def g(i, **kwargs):
        n.add_node(hgnc(i), dict(color=hgnc.color, annotations=kwargs))

    def p(i):
        n.add_node(up(i), dict(color=up.color))

    def m(i, **kwargs):
        n.add_node(mi(i), dict(color=mi.color, annotations=kwargs))

    def e(i, j, *args, **kwargs):
        n.add_edge(i, j, *args, **kwargs)

    return g, p, m, e


def generate_example_1():
    """
    Generates a small, three-class heterogenous network with labels corresponding to
    genes, proteins, and miRNAs. This example is appropriate for testing edge induction.
    """

    shn = HetNet(params={
        hgnc.color: {
            'regulation': [DOWN, UP, UNREGULATED]
        },
        up.color: {},
        mi.color: {
            'c1': ['a', 'b', 'c'],
            'c2': ['d', 'e', 'f']
        }
    })

    g, p, m, e = quick_adders(shn)

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

    for i in range(1, 17):
        p(i)
        e(hgnc(i), up(i))

    m(17, c1='a', c2='f')
    m(18, c1='a', c2='e')
    m(19, c1='n', c2='d')
    m(20, c1='n', c2='e')

    e(mi(17), hgnc(17))
    e(mi(18), hgnc(18))
    e(mi(19), hgnc(19))
    e(mi(20), hgnc(20))

    e(up(3), mi(17))
    e(up(9), mi(17))
    e(up(4), mi(17))
    e(up(7), mi(18))
    e(up(16), mi(19))
    e(up(7), mi(19))
    e(up(5), mi(20))
    e(up(13), mi(20))

    e(up(1), up(2))
    e(up(2), up(6))
    e(up(5), up(6))
    e(up(6), up(7))
    e(up(7), up(12))
    e(up(6), up(12))
    e(up(12), up(13))
    e(up(12), up(14))
    e(up(14), up(15))
    e(up(7), up(10))
    e(up(10), up(11))
    e(up(11), up(16))
    e(up(8), up(10))
    e(up(8), up(9))
    e(up(9), up(10))
    e(up(3), up(8))
    e(up(3), up(9))
    e(up(3), up(4))

    e(hgnc(5), hgnc(13))
    e(hgnc(1), hgnc(2))
    e(hgnc(5), hgnc(6))
    e(hgnc(3), hgnc(4))
    e(hgnc(4), hgnc(8))
    e(hgnc(8), hgnc(9))
    e(hgnc(8), hgnc(10))
    e(hgnc(10), hgnc(16))
    e(hgnc(7), hgnc(10))
    e(hgnc(15), hgnc(19))

    return shn


# @cli.command()
def generate_example_2():
    """
    Generates example appropriate for testing footprint calculations
    :return:
    """
    shn = HetNet(params={
        'r': {},
        'b': {},
        'g': {}
    })

    shn.add_node('r1', dict(color='r'))
    shn.add_node('r2', dict(color='r'))
    shn.add_node('b1', dict(color='b'))
    shn.add_node('b2', dict(color='b'))
    shn.add_node('g1', dict(color='g'))

    shn.add_edge('r1', 'r2')
    shn.add_edge('r1', 'g1')
    shn.add_edge('r1', 'b1')
    shn.add_edge('r2', 'b1')
    shn.add_edge('b1', 'b2')
    shn.add_edge('b2', 'g1')

    return shn


def generate_example_3():
    """
    Generates example appropriate for testing entropy calculations
    :return:
    """
    shn = HetNet(params={
        'r': {},
        'b': {},
        'g': {}
    })

    shn.add_node('r1', dict(color='r'))
    shn.add_node('r2', dict(color='r'))
    shn.add_node('b1', dict(color='b'))
    shn.add_node('b2', dict(color='b'))
    shn.add_node('b3', dict(color='b'))
    shn.add_node('b4', dict(color='b'))
    shn.add_node('g1', dict(color='g'))
    shn.add_node('g2', dict(color='g'))
    shn.add_node('g3', dict(color='g'))

    shn.add_edge('r1', 'b1')
    shn.add_edge('r1', 'b2')
    shn.add_edge('r1', 'b4')
    shn.add_edge('r2', 'b2')
    shn.add_edge('r2', 'b3')
    shn.add_edge('b1', 'g1')
    shn.add_edge('b2', 'g1')
    shn.add_edge('b4', 'g3')
    shn.add_edge('b3', 'g2')
    shn.add_edge('b3', 'b2')

    return shn


def generate_example_4():
    shn = HetNet(params={
        hgnc.color: {},
        up.color: {},
        mi.color: {}
    })

    shn.add_node(hgnc(1), dict(color=hgnc.color))
    shn.add_node(hgnc(2), dict(color=hgnc.color))
    shn.add_node(hgnc(3), dict(color=hgnc.color))
    shn.add_node(hgnc(4), dict(color=hgnc.color))
    shn.add_node(hgnc(5), dict(color=hgnc.color))

    shn.add_node(up(1), dict(color=up.color))
    shn.add_node(up(2), dict(color=up.color))
    shn.add_node(up(3), dict(color=up.color))
    shn.add_node(up(4), dict(color=up.color))

    shn.add_node(mi(1), dict(color=mi.color))

    shn.add_edge(hgnc(2), hgnc(4))

    shn.add_edge(hgnc(1), up(1))
    shn.add_edge(hgnc(2), up(2))
    shn.add_edge(hgnc(3), up(3))
    shn.add_edge(hgnc(4), up(4))

    shn.add_edge(up(1), up(2))
    shn.add_edge(up(1), up(3))
    shn.add_edge(up(2), up(3))
    shn.add_edge(up(3), up(4))

    shn.add_edge(hgnc(5), mi(1))

    shn.add_edge(mi(1), up(2))
    shn.add_edge(mi(1), up(3))

    return shn


def generate_example_5():
    """
    Generates a small example binary network
    """

    shn = HetNet(params={
        'red': {},
        'yellow': {}
    })

    shn.add_node('a', dict(color='red'))
    shn.add_node('b', dict(color='red'))
    shn.add_node('c', dict(color='red'))
    shn.add_node('d', dict(color='red'))
    shn.add_node('e', dict(color='red'))

    shn.add_node('1', dict(color='yellow'))
    shn.add_node('2', dict(color='yellow'))
    shn.add_node('3', dict(color='yellow'))
    shn.add_node('4', dict(color='yellow'))
    shn.add_node('5', dict(color='yellow'))

    rr = [("a", "b"), ("b", "c"), ("d", "e")]
    ry = [("1", "d"), ("2", "c"), ("3", "a")]
    yy = [("1", "2"), ("2", "3"), ("1", "3"), ("3", "4"), ("4", "5")]

    for u, v in chain(rr, ry, yy):
        shn.add_edge(u, v)

    return shn
