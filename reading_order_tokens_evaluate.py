import argparse
import os
import pickle

import torch

from language_model.carrier import default_model, default_vocab
from document import page_xml
from reading_order.metric.prima import compare, calculate_penalty, calculate_penalty_percentage
from reading_order.processor import Processor

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True,
                    help='Folder with xml files to create reading order and compare')
parser.add_argument('--save', type=str, required=True, help='Results will be save to file')
parser.add_argument('--cuda', help='use CUDA', action='store_true')

args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if not os.path.isdir(args.folder):
    print('Folder not found')
    exit()

path = os.path.abspath(args.folder)
save = os.path.abspath(args.save)

files = []
for file in os.listdir(os.path.abspath(path)):
    if file.endswith(".document"):
        files.append(file)


files = sorted(files)

lengths = list(range(1, 11)) + list(range(12, 33, 2)) + [-1]
results = []
model = default_model()
model.to(torch.device("cuda" if args.cuda else "cpu"))
processor = Processor(model, default_vocab())

for file in files:
    filepath = os.path.join(path, file)
    xml = page_xml.parse(filepath)

    for length in lengths:
        print('{} - {}'.format(file, length), end='')

        if length == -1:
            length = None

        identified_ro = processor.process(xml.get_text_regions(), token_limit=length)
        cmp = compare(xml.get_reading_order(), identified_ro)

        results.append({
            'file': file,
            'length': length,
            'cmp': cmp,
            'penalty': calculate_penalty(cmp),
            'penalty_percentage': calculate_penalty_percentage(cmp, xml)
        })

        print(' done!')
        processor.clear()

with open(save, 'wb') as f:
    pickle.dump(results, f)