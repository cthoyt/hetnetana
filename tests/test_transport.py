from hetnetana.struct import hetnet_examples, multihetnet_examples
from hetnetana.struct.transport import convert
from tests.utils import HetnetanaTestBase


class TestTransport(HetnetanaTestBase):
    def test_convert(self):
        simple_hetnet = hetnet_examples.generate_example_4()

        collapse_keys = {
            'm': 'mti',
            'p': 'ppi'
        }

        edge_keys = {
            'g': {
                'g': 'coexp',
                'm': 'encode'
            }
        }
        calculated_multihetnet = convert(simple_hetnet, collapse_color='p', collapse_source='g',
                                         collapse_keys=collapse_keys, edge_annotations=edge_keys)

        target_multihetnet = multihetnet_examples.generate_example_4()

        self.assertCountEqual(target_multihetnet.nodes(), calculated_multihetnet.nodes())
        self.assertGraphEqual(target_multihetnet, calculated_multihetnet)
