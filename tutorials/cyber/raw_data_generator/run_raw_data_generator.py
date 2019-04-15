import argparse
from raw_data_generator import raw_data_generator

def main(input_file, out_dir, out_format):
    rdg = raw_data_generator.RawDataGenerator()
    rdg.generate_raw_data(input_file, out_dir, out_format)

if __name__ == '__main__':
    # Read commandline arguments
    PARSER = argparse.ArgumentParser(description='LANL Raw Data Generator')
    PARSER.add_argument('input_file', help='Full path to input data source file')
    PARSER.add_argument('output_directory', help='Full path to desired output directory')
    PARSER.add_argument('--format', choices=['csv', 'json'], default='csv', type=str)
    args = PARSER.parse_args()
    main(args.input_file, args.output_directory, args.format)
