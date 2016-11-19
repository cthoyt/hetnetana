from .struct.hetnet import HetNet
from .struct.multihetnet import MultiHetNet

__version__ = '0.1.0'

__title__ = 'hetnetana'
__description__ = 'A Python package for integrating data and performing topological footprint analysis'
__url__ = 'https://tor-2.scai.fraunhofer.de/gf/project/banana/'

__author__ = 'Charles Tapley Hoyt'
__email__ = 'charles.hoyt@scai.fraunhofer.de'

__license__ = 'All Rights Reserved.'
__copyright__ = 'Copyright (c) 2016 Charles Tapley Hoyt'

__all__ = ['HetNet', 'MultiHetNet', 'hgnc', 'mi', 'up']

COLOR = 'color'
ANNOTATIONS = 'annotations'
TYPE = 'type'


class ColorFormatter:
    def __init__(self, fmt, color):
        self.fmt = fmt
        self.color = color

    def __call__(self, i):
        return self.fmt.format(i)

    def __str__(self):
        return self.color


hgnc = ColorFormatter('hgnc{}', 'g')
mi = ColorFormatter('MI{:07}', 'm')
up = ColorFormatter('UP{:04}', 'p')
snp = ColorFormatter('rs{:07}', 's')
