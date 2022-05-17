import argparse
import os

import torch

from language_model.carrier import cs_model, cs_vocab
from document import page_xml
from reading_order.metric.prima import compare
from language_model.analyzer import LmAnalyzer
from reading_order.metric.recall import compare as dp_compare

"""
Skript pro jazykovou analyzu. Soubor musi mit ground truth pro porovnani.
Vyuzito predevsim za ucelem implementace
"""

parser = argparse.ArgumentParser(description='Reading order process')
parser.add_argument('--file', type=str, required=True)

args = parser.parse_args()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.file):
    print('Data not found')
    exit()

xml = page_xml.parse(args.file)
processor = LmAnalyzer(cs_model(device), cs_vocab())
identified_ro = processor.analyze(xml)
cmp = compare(xml, xml.get_reading_order(), identified_ro)

print('penalty: {}'.format(cmp.penalty()))
print('penalty_percentage: {}'.format(cmp.percentage()))

dp_results = dp_compare(xml.get_reading_order(), identified_ro)
print('total: {}'.format(dp_results.total()))
print('hits: {}'.format(dp_results.hits()))
print('missed: {}'.format(dp_results.missed()))
print('recall: {}%'.format(dp_results.recall()))