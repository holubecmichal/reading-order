import argparse
import os

from document.page_xml import parse
from language_model.carrier import de_vocab, cs_vocab

"""
Skript pro vypocet statistik PageXML souboru
--path je cesta ke slozce s temito soubory nebo konkretni soubor
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str, required=True)

args = parser.parse_args()
path = os.path.abspath(args.path)

if os.path.isfile(path):
    files = [os.path.basename(path)]
    dir = os.path.dirname(path)
else:
    dir = path
    files = []

    for filename in os.listdir(path):
        if not filename.endswith('.xml'):
            continue

        files.append(filename)

all_regions = 0
text_regions = 0
length = 0
tokens = 0
unks = 0

vocab = cs_vocab()
for file in files:
    filepath = os.path.join(dir, file)
    doc = parse(filepath)
    r = doc.get_text_regions()

    all_regions += len(doc.get_regions())
    text_regions += len(doc.get_text_regions())
    lenghts = [len(r[x].get_text()) for x in r]
    length += sum(lenghts) / len(doc.get_text_regions())

    for t in vocab.Encode([r[x].get_text() for x in r]):
        if t.count(0):
            unks += 1



files_count = len(files)

avg_unks = unks / text_regions
all_regions /= files_count
avg_text_regions = text_regions / files_count
length /= files_count

print('files: {}, avg regions: {}, text regions: {} avg text regions: {}, avg text length: {}, unks avg: {}, unks: {}'.format(
    len(files), round(all_regions), text_regions, round(avg_text_regions), round(length), round(avg_unks, 2), unks))
