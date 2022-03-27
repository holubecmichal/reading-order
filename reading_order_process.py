import argparse
import os

from language_model.carrier import default_model, default_vocab
from document import page_xml
from reading_order.metric.prima import compare, calculate_penalty, calculate_penalty_percentage
from reading_order.processor import Processor

parser = argparse.ArgumentParser(description='Reading order process')
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--tokens', type=int, required=True)

args = parser.parse_args()

if not os.path.exists(args.file):
    print('Data not found')
    exit()

xml = page_xml.parse(args.file)
processor = Processor(default_model(), default_vocab())
processor.process_print = True

identified_ro = processor.process(xml.get_text_regions(), token_limit=args.tokens)
cmp = compare(xml.get_reading_order(), identified_ro)

print('penalty: {}'.format(calculate_penalty(cmp)))
print('penalty_percentage: {}'.format(calculate_penalty_percentage(cmp, xml)))

