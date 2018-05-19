import sys
import argparse
import os
from waldo.egs.madcat_arabic.v1.scoring.scoring_utils import get_score

parser = argparse.ArgumentParser(
    description='scoring script for text localization')
parser.add_argument('--hypothesis', default='data/exp/', type=str,
                    help='hypothesis directory of test data')
parser.add_argument('--reference', default='data/test', type=str,
                    help='reference directory of test data')
args = parser.parse_args()


def main():
    reference_handle = open(args.reference, 'r')
    reference_data = reference_handle.read().strip().split('\n')

    hypothesis_handle = open(args.hypothesis, 'r')
    hypothesis_data = hypothesis_handle.read().strip().split('\n')

    score = get_score(reference_data, hypothesis_data)

if __name__ == '__main__':
      main()

