import argparse
import configparser
from raw_data_generator import raw_data_generator
from hdfs3 import HDFileSystem

def main(input_file, out_dir, out_format, hdfs_config):
    hdfs_conn = HDFileSystem(host = hdfs_config['host'], port=int(hdfs_config['port']), user=hdfs_config['user'])
    rdg = raw_data_generator.RawDataGenerator(hdfs_conn)
    rdg.generate_raw_data(input_file, out_dir, out_format)

if __name__ == '__main__':
    # Read commandline arguments
    PARSER = argparse.ArgumentParser(description='LANL Raw Data Generator')
    PARSER.add_argument('input_file', help='Full path to hdfs input data source file')
    PARSER.add_argument('output_directory', help='Full path to hdfs desired output directory')
    PARSER.add_argument('--format', choices=['csv', 'json'], default='csv', type=str)
    args = PARSER.parse_args()
    config = configparser.ConfigParser()
    config.read('run_raw_data_generator.ini')
    main(args.input_file, args.output_directory, args.format, config['HDFS'])
