import argparse

parser = argparse.ArgumentParser(description='Set up paths for decoding scripts')
parser.add_argument('base_dir', type=str, help='Full path to base directory where all outputs will be stored.')
args = parser.parse_args()

base_dir = args.base_dir

