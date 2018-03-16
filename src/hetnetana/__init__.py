from .constants import COLOR, ANNOTATIONS, TYPE
from .struct.hetnet import HetNet
from .struct.multihetnet import MultiHetNet

__all__ = ['HetNet', 'MultiHetNet', 'hgnc', 'mi', 'up']

__version__ = '0.1.0'

__title__ = 'hetnetana'
__description__ = 'A Python package for integrating data and performing topological footprint analysis'
__url__ = 'https://github.com/cthoyt/hetnetana'

__author__ = 'Charles Tapley Hoyt'
__email__ = 'charles.hoyt@scai.fraunhofer.de'

__license__ = 'All Rights Reserved.'
__copyright__ = 'Copyright (c) 2016-2018 Charles Tapley Hoyt'




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
