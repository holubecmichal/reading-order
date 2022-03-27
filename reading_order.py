import argparse
import os

from language_model.carrier import default_model, default_vocab
from document import page_xml
from reading_order.processor import Processor

parser = argparse.ArgumentParser(description='Reading order')
parser.add_argument('--data', type=str)

args = parser.parse_args()

if not args.data or not os.path.exists(args.data):
    print('Data not found')
    exit()

xml = page_xml.parse(args.data)
candidates = xml.get_text_regions()
processor = Processor(default_model(), default_vocab())

# test = {}
# while len(test) < 10:
#     iid, candidate = candidates.popitem()
#     test[iid] = candidate
#
# candidates = test
processor.process(candidates)