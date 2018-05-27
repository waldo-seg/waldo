import argparse
import os
from waldo.egs.madcat_arabic.v1.local.scoring.scoring_utils import get_score

parser = argparse.ArgumentParser(
    description='scoring script for text localization')
parser.add_argument('--hypothesis', default='/Users/ashisharora/google_Drive/madcat_arabic/scoring/hyp', type=str,
                    help='hypothesis directory of test data')
parser.add_argument('--reference', default='/Users/ashisharora/google_Drive/madcat_arabic/scoring/ref', type=str,
                    help='reference directory of test data')
args = parser.parse_args()


def main():
    reference_handle = open(os.path.join(args.reference, '1.txt'), 'r')
    ref_file = reference_handle.read().strip().split('\n')

    hypothesis_handle = open(os.path.join(args.hypothesis, '1.txt'), 'r')
    hyp_file = hypothesis_handle.read().strip().split('\n')

    print(get_score(ref_file, hyp_file))


if __name__ == '__main__':
      main()

