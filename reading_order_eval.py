import argparse
import os.path
import pickle

import torch

from document.page_xml import parse
from language_model.carrier import de_model, de_vocab, cs_model, cs_vocab
from language_model.analyzer import LmAnalyzer as LmAnalyzer
from spatial.analyzer import DiagonalAnalyzer, ColumnarAnalyzer, ColumnarLmAnalyzer

"""
Skript pro vyhodnoceni experimentu
--path je cesta do slozky s xml dokumenty, pripadne na konkretni xml soubor

Vysledky jsou serializovany a ulozeny do pkl souboru
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str, required=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
path = os.path.abspath(args.path)

model = cs_model(device)
vocab = cs_vocab()

# model = de_model(device)
# vocab = de_vocab()

errors = []


def save(dir, file, content: dict):
    filename = file.replace('.xml', '') + '.pkl'
    with open(os.path.join(dir, filename), 'wb') as f:
        pickle.dump(content, f)


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

if not files:
    print('Not files specified')
    exit()

for i, file in enumerate(sorted(files, reverse=True)):
    print("{}: {}".format(i, file))

    try:
        filepath = os.path.join(dir, file)

        if os.path.isfile(filepath.replace('xml', 'pkl')):
            continue

        doc = parse(filepath)

        lmAnalyzer = LmAnalyzer(model, vocab)
        lmAnalyzer.process_print = True

        diagonalAnalyzer = DiagonalAnalyzer()
        columnarAnalyzer = ColumnarAnalyzer()
        columnarLmAnalyzer = ColumnarLmAnalyzer()

        results = {}

        print('lm-h-2')
        lmAnalyzer.use_hard_limit(2)
        results['lm-H-2'] = lmAnalyzer.analyze(doc)

        print('lm-h-3')
        lmAnalyzer.use_hard_limit(3)
        results['lm-H-3'] = lmAnalyzer.analyze(doc)

        print('lm-s-4')
        lmAnalyzer.use_score_hard_limit(4)
        results['lm-S-4'] = lmAnalyzer.analyze(doc)

        print('lm-s-5')
        lmAnalyzer.use_score_hard_limit(5)
        results['lm-S-5'] = lmAnalyzer.analyze(doc)

        print('lm-s-6')
        lmAnalyzer.use_score_hard_limit(6)
        results['lm-S-6'] = lmAnalyzer.analyze(doc)

        results['diag'] = diagonalAnalyzer.analyze(doc)
        results['col'] = columnarAnalyzer.analyze(doc)

        lmAnalyzer.use_hard_limit(2)
        results['comb-H-2'] = columnarLmAnalyzer.analyze(doc, lmAnalyzer)

        lmAnalyzer.use_hard_limit(3)
        results['comb-H-3'] = columnarLmAnalyzer.analyze(doc, lmAnalyzer)

        lmAnalyzer.use_score_hard_limit(4)
        results['comb-S-4'] = columnarLmAnalyzer.analyze(doc, lmAnalyzer)

        lmAnalyzer.use_score_hard_limit(5)
        results['comb-S-5'] = columnarLmAnalyzer.analyze(doc, lmAnalyzer)

        lmAnalyzer.use_score_hard_limit(6)
        results['comb-S-6'] = columnarLmAnalyzer.analyze(doc, lmAnalyzer)

        torch.cuda.empty_cache()
        save(dir, file, results)
    except Exception as e:
        raise e
        s = str(e)
        err = "{} - {}".format(file, s)
        errors.append(err)
        print(err)


for err in errors:
    print(err)
