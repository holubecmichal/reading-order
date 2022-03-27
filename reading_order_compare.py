import argparse
import os

from document import page_xml
from reading_order.metric.prima import compare

parser = argparse.ArgumentParser(description='Reading order compare')
parser.add_argument('--ground_truth', type=str, required=True)
parser.add_argument('--actual', type=str, required=True)

args = parser.parse_args()

if not os.path.exists(args.ground_truth) or not os.path.exists(args.actual):
    print('Data not found')
    exit()

ground_truth_xml = page_xml.parse(args.ground_truth)
actual_xml = page_xml.parse(args.actual)

ground_truth = ground_truth_xml.get_reading_order()
actual = actual_xml.get_reading_order()
results = compare(ground_truth_xml, ground_truth, actual)

results.print()
print('penalty: {}'.format(results.penalty()))
print('penalty_percentage: {}'.format(results.percentage()))