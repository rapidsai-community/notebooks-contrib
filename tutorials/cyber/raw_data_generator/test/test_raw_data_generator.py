import unittest
import os
from raw_data_generator import raw_data_generator
import sys
from mockito import mock, when

class RawDataGeneratorTest(unittest.TestCase):

    def setUp(self):
        sys.modules['HDFileSystem'] = mock()

    def test_data_generator(self):
        cwd = os.getcwd()
        output_dir = "/data/lanl/output"
        input_dir = "/test/data/input"
        input_path = cwd + input_dir + "/sample_data.txt"
        output_path = cwd + output_dir + "/sample_data.csv"
        hdfs_conn = mock()
        when(hdfs_conn).open(input_path, 'rb').thenReturn(open(input_path, 'rb'))
        if not os.path.exists(cwd + output_dir):
            os.makedirs(cwd + output_dir)
        when(hdfs_conn).open(output_path, 'wb').thenReturn(open(output_path, 'wb'))
        rdg = raw_data_generator.RawDataGenerator(hdfs_conn)
        rdg.generate_raw_data(input_path, cwd+output_dir, 'csv')
        assert(os.path.isfile(input_path))

