import json
import sys

import click
import networkx as nx

from .analysis import main as analysis_main
from .generation.generate_toy import main as generate_toy_main
from .struct.hetnet import HetNet
from .struct.hetnet_examples import generate_example_1, generate_example_2, generate_example_3
from .struct.multihetnet_io import from_resource as multihetnet_from_resource


@click.group()
def main():
    pass


@main.command()
@click.option('--directory', '-d', type=click.Path(), required=True)
@click.option('--percent', '-p', type=float)
@click.option('--seed', '-seed', type=int)
def generate_toy(directory, percent=None, seed=None):
    generate_toy_main(directory, percent, seed)


@main.command()
@click.option('--resources', '-r', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path())
@click.option('--multihetnet', is_flag=True)
def make_gml(resources, output, multihetnet):
    if multihetnet:
        hn = multihetnet_from_resource(resources)
    else:
        hn = HetNet.from_resource(resources)

    nx.write_gml(hn, output)


@main.command()
@click.option('--resources', '-r', type=click.Path(exists=True), required=True)
@click.option('--output', '-o', type=click.File('w'), default=sys.stdout)
@click.option('--parameters', '-p', type=click.File('r'))
@click.option('--nodes', type=click.File('r'))
@click.option('--entropies', is_flag=True)
@click.option('--normalize', is_flag=True)
@click.option('--stochastic', is_flag=True)
@click.option('--multihetnet', is_flag=True)
def footprints(resources, output, parameters, nodes, entropies, normalize, stochastic, multihetnet):
    if not multihetnet:
        hn = HetNet.from_resource(resources)
    else:
        hn = multihetnet_from_resource(resources)

    kwargs = {}

    if nodes is not None:
        n = []
        for line in nodes:
            sline = line.strip()
            if sline in hn:
                n.append(sline)
        kwargs['nodes'] = n

    if parameters is not None:
        kwargs.update(json.load(parameters))

    features = hn.calculate_footprints(stochastic=stochastic, normalize=normalize, entropies=entropies, **kwargs)
    features.to_csv(output)


def jsontr_helper(d, keys):
    it = iter(keys)
    result = d[next(it)]
    for key in it:
        if isinstance(result, list):
            key = int(key)
        result = result[key]
    return result


@main.command()
@click.argument('file', type=click.File('r'))
@click.argument('keys', nargs=-1)
@click.option('--maintain', '-k', is_flag=True)
@click.option('--output', '-o', default=sys.stdout, type=click.File('w'))
def jsontr(file, keys, maintain, output):
    result = jsontr_helper(json.load(file), keys)

    if maintain:
        json.dump(result, output)
    elif isinstance(result, list):
        for el in result:
            print(el, file=output)
    elif isinstance(result, dict):
        for k, v in result.items():
            print(k, v, sep='\t', file=output)
    else:
        print(result, file=output)


@main.command()
@click.argument('directory')
@click.argument('example', type=click.Choice([1, 2, 3]), default=1)
def generate_example(directory, example):
    if 3 == example:
        generate_example_3().to_resource(directory)
    elif 2 == example:
        generate_example_2().to_resource(directory)
    else:
        generate_example_1().to_resource(directory)


@main.command()
@click.argument('footprints')
@click.argument('training')
@click.argument('directory')
def analysis(footprints, training, directory):
    analysis_main(footprints, training, directory)


if __name__ == '__main__':
    main()
