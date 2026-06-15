"""Convert a .npy output file to CSV format.

Usage:
    python ToCsv.py --input output.npy --output out.csv
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Convert .npy output to CSV')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input .npy file')
    parser.add_argument('--output', type=str, default='out.csv',
                        help='Path to output CSV file')
    args = parser.parse_args()

    data = np.load(args.input)
    cleaned = data[~np.isnan(data)]
    np.savetxt(args.output, [cleaned], delimiter=',')
    print(f'Saved {len(cleaned)} values to {args.output}')


if __name__ == '__main__':
    main()
