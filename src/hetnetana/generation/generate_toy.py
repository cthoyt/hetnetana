#!/usr/bin/env python

import itertools as itt
import logging
import os
from datetime import datetime
from getpass import getuser

import numpy as np
import pandas as pd

from .generate import get_percentile_diff, get_inducible_pairs
from .. import hgnc, mi, up, snp as rs
from ..struct.hetnet import HetNet, encode_color_path

log = logging.getLogger()

CP1 = ('p', 'm')
CP1_TERMINAL = ('p', ('m', ()))
CP1_STR = encode_color_path(CP1_TERMINAL, color_path_type='terminal')
CP2 = ('p', 'p', ('g', (('regulated', 'T'),)))
CP2_STR = encode_color_path(CP2, color_path_type='terminal')


def convert_simple_to_terminal(cp):
    *head, tail = cp
    tail = (tail, ())

    if isinstance(cp, list):
        return list(head) + [tail]

    return tuple(head) + (tail,)


def generate_toy(seed=None):
    np.random.seed(seed)
    h = HetNet({
        "g": {
            "regulated": ["T", "F"]  # significant regulation
        },
        "p": {
        },
        "m": {
        },
        "s": {
        }
    })

    n_genes = 426
    n_proteins = 275
    n_mirnas = 16
    n_snps = 14

    n_mirna_encoding_genes = 8
    n_protein_encoding_genes = 275

    n_ppis = 292
    n_mtis = 51

    # ??? not sure about numbers
    n_regulated = 25
    n_coexprs = 400

    regulated = np.random.choice(np.arange(n_genes), size=n_regulated, replace=False)
    genes = {}
    for i in range(n_genes):
        gene = hgnc(i)
        genes[i] = gene
        h.add_node(gene, dict(color='g', annotations={'regulated': ("T" if i in regulated else "F")}))

    protein_encoding_genes = sorted(genes.values())[:n_protein_encoding_genes]
    proteins = {}
    for i, gene in zip(range(n_proteins), protein_encoding_genes):
        protein = up(i)
        proteins[i] = protein
        h.add_node(protein, dict(color='p', annotations={}))
        h.add_edge(gene, protein)

    mirna_encoding_genes = list(
        np.random.choice(list(set(genes.values()) - set(protein_encoding_genes)), size=n_mirna_encoding_genes,
                         replace=False))
    mirnas = {}
    for i, gene in zip(range(n_mirnas), 2 * mirna_encoding_genes):
        mirna = mi(i)
        mirnas[i] = mirna
        h.add_node(mirna, dict(color='m', annotations={}))
        h.add_edge(gene, mirna)

    h.mirna_encoding = mirna_encoding_genes

    mutations = np.random.choice(list(set(genes.values()) - set(protein_encoding_genes) - set(mirna_encoding_genes)),
                                 size=n_snps, replace=False)
    snps = {}
    for i, gene in zip(range(n_snps), mutations):
        snp = rs(i)
        snps[i] = snp
        h.add_node(snp, dict(color='s', annotations={}))
        h.add_edge(gene, snp)

    pp = list(itt.combinations(proteins, 2))
    for i in np.random.choice(len(pp), size=n_ppis, replace=False):
        a, b = pp[i]
        h.add_edge(proteins[a], proteins[b])

    gg = list(itt.combinations(genes, 2))
    for i in np.random.choice(len(gg), size=n_coexprs, replace=False):
        a, b = gg[i]
        h.add_edge(genes[a], genes[b])

    mp = list(itt.product(proteins, mirnas))
    for i in np.random.choice(len(mp), size=n_mtis, replace=False):
        a, b = mp[i]
        h.add_edge(proteins[a], mirnas[b])

    h.graph['generation_manifest'] = {
        'user': getuser(),
        'generation_time': str(datetime.now()),
        'np_random_seed': seed,
        'protein_encoding': protein_encoding_genes
    }

    return h


def induce_toy(graph, target_nodes, upper=100, lower=90, loc_coef=1.1, scale_coef=1.2, seed=None):
    """
    Generate network pertaining to actual biology, and induce 2 features over protein-coding genes:
        - protein-mirna
        - protein-protein-regulated_gene

    :param graph: the network to induce
    :param target_nodes: the nodes to induce in the network
    :param upper: the upper percentile to calculate for the color paths
    :param lower: the lower percentile to calculate for the color paths
    :param loc_coef: the factor to multiply the upper percentile for induction for the mean of random gaussian sampling
    :param scale_coef: the factor to multiply the upper-lower percentile difference for the standard deviation of random
                        gaussian sampling
    :param seed: seed for numpy random number generator
    :return:
    """

    np.random.seed(seed)
    pairs = []

    # TODO factor out induction parameters
    p100a, p98a, pt2a = get_percentile_diff(graph, CP1, upper, lower, color_path_type='simple')
    log.debug(
        'CP {} -> {}%: {}, {}%: {}, D{}%: {}'.format(encode_color_path(CP1, color_path_type="simple"), upper, p100a,
                                                     lower, p98a, upper - lower, pt2a))

    p100b, p98b, pt2b = get_percentile_diff(graph, CP2, upper, lower, color_path_type='terminal')
    log.debug(
        'CP {} -> {}%: {}, {}%: {}, D{}%: {}'.format(encode_color_path(CP2, color_path_type="terminal"), upper, p100b,
                                                     lower, p98b, upper - lower, pt2b))

    for node in target_nodes:
        for edge in get_inducible_pairs(graph, node, CP1,
                                        int(np.random.normal(loc=(loc_coef * p100a), scale=(scale_coef * pt2a))),
                                        color_path_type='simple'):
            pairs.append(edge)
        for edge in get_inducible_pairs(graph, node, CP2,
                                        int(np.random.normal(loc=(loc_coef * p100b), scale=(scale_coef * pt2b))),
                                        color_path_type='terminal'):
            pairs.append(edge)

    """
    params = [
        (CP1, upper, lower, 1.1, 1.2, 'simple'),
        (CP2, upper, lower, 1.1, 1.2, 'terminal')
    ]
    pairs = []
    for cp, up, lo, lc, sc, cpt in params:
        p100, p98, pt2 = get_percentile_diff(graph, cp, up, lo, color_path_type=cpt)
        for node in target_nodes:
            ne = int(np.random.normal(loc=(lc * p100), scale=(sc * pt2)))
            pairs.extend(get_inducible_pairs(graph, node, cp, ne, color_path_type=cpt))
    """

    for a, b in set(pairs):
        graph.add_edge(a, b)

    graph.graph['induction_manifest'] = {
        'user': getuser(),
        'induction_time': str(datetime.now()),
        'induced': sorted(target_nodes),
        'upper': upper,
        'lower': lower,
        'loc_coef': loc_coef,
        'scale_coef': scale_coef,
        'np_random_seed': seed
    }

    return graph


def main(directory, percent=0.8, seed=None):
    """
    :param directory: output directory
    :param percent: if given, outputs a training and test manifest
    :param seed: seed for numpy random number generator
    """

    np.random.seed(seed)

    h = generate_toy()
    n_induce = int(0.5 + 7 / percent)
    target_nodes = np.random.choice(h.graph['generation_manifest']['protein_encoding'], size=n_induce, replace=False)
    hn = induce_toy(h, target_nodes)
    hn.to_resource(directory)

    induced = hn.graph['induction_manifest']['induced']
    nodes = sorted(hn.graph['generation_manifest']['protein_encoding'])

    full_induction_manifest = pd.DataFrame([node in induced for node in nodes], index=nodes, columns=['induced'])
    full_induction_manifest.to_csv(os.path.join(directory, 'full_induce_manifest.csv'))

    not_induced = list(set(nodes) - set(induced))
    n_induced = len(induced)
    n_not_induced = len(not_induced)

    np.random.shuffle(induced)
    np.random.shuffle(not_induced)

    head_induced = induced[:int(percent * n_induced)]
    tail_induced = induced[int(percent * n_induced):]

    head_not_induced = not_induced[:int(percent * n_not_induced)]
    tail_not_induced = not_induced[int(percent * n_not_induced):]

    head = sorted(head_induced) + sorted(head_not_induced)
    tail = sorted(tail_induced) + sorted(tail_not_induced)

    full_induction_manifest.loc[head].to_csv(os.path.join(directory, 'training_induce_manifest.csv'))
    full_induction_manifest.loc[tail].to_csv(os.path.join(directory, 'test_induce_manifest.csv'))
