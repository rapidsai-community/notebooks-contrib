import unittest
import os
from raw_data_generator import raw_data_generator

class RawDataGeneratorTest(unittest.TestCase):

    def test_data_generator(self):
        output_dir = "/data/lanl/output"
        input_path = os.getcwd() + "/test/data/input/sample_data.txt"
        output_path = output_dir + "/sample_data.csv"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        rdg = raw_data_generator.RawDataGenerator()
        rdg.generate_raw_data(input_path, output_dir, 'csv')
        assert(os.path.isfile(output_path))

