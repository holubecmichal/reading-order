import argparse
import os
import pickle

from document import page_xml
from reading_order.metric.prima import compare, calculate_penalty, calculate_penalty_percentage

parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, required=True,
                    help='Folder with ground truths files')
parser.add_argument('--tb', type=str, required=True,
                    help='Folder with top-to-bottom files')
parser.add_argument('--ocr', type=str, required=True,
                    help='Folder with ground truths files')
parser.add_argument('--save', type=str, required=True, help='Results will be save to file')

args = parser.parse_args()

if not os.path.isdir(args.gt):
    print('--gt folder not found')
    exit()

if not os.path.isdir(args.tb):
    print('--tb folder not found')
    exit()

if not os.path.isdir(args.ocr):
    print('--ocr folder not found')
    exit()

gt_folder = os.path.abspath(args.gt)
tb_folder = os.path.abspath(args.tb)
ocr_folder = os.path.abspath(args.ocr)
save = os.path.abspath(args.save)

files = []
for file in os.listdir(gt_folder):
    if file.endswith(".xml"):
        files.append(file)

files = sorted(files)
results = []

for file in files:
    print(file, end='')

    gt_file = os.path.join(gt_folder, file)
    tb_file = os.path.join(tb_folder, file)
    ocr_file = os.path.join(ocr_folder, file)

    gt = page_xml.parse(gt_file)
    tb = page_xml.parse(tb_file)
    ocr = page_xml.parse(ocr_file)

    cmp = compare(gt.get_reading_order(), tb.get_reading_order())
    results.append({
        'file': file,
        'type': 'tb',
        'cmp': cmp,
        'penalty': calculate_penalty(cmp),
        'penalty_percentage': calculate_penalty_percentage(cmp, gt)
    })

    cmp = compare(gt.get_reading_order(), ocr.get_reading_order())
    results.append({
        'file': file,
        'type': 'ocr',
        'cmp': cmp,
        'penalty': calculate_penalty(cmp),
        'penalty_percentage': calculate_penalty_percentage(cmp, gt)
    })

    print(' done!')


asd = 0

with open(save, 'wb') as f:
    pickle.dump(results, f)