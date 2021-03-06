import argparse
import os
import pickle

from matplotlib import pyplot as plt

from document.page_xml import parse
from reading_order.metric.recall import compare as dp_compare
from reading_order.metric.prima import compare as prima_compare
from utils.collection import Collection

"""
Skript pro vyhodnoceni vysledku experimentu
--path je cesta ke slozce s pkl soubory nebo ke konkretnimu souboru

Skript pripravi a do pdf ulozi vysledky pro jednotlive pkl soubory a nakonec provede i agregaci vsech souboru
pro ktery vytvori souhrnny graf
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
        if not filename.endswith('.pkl'):
            continue

        files.append(filename)

if not files:
    print('Not files specified')
    exit()

results = {}
methods = []

for file in sorted(files):
    print(file)
    filepath = os.path.join(dir, file)
    base_filename = file.replace('.pkl', '')
    original = parse(os.path.join(dir, base_filename + '.xml'))
    ro = original.get_reading_order()
    results[file] = {}

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(e)
        continue

    for method in data:
        ro = data[method]

        if method not in methods:
            methods.append(method)

        results[file][method] = {
            'prima': prima_compare(original, original.get_reading_order(), ro).percentage(),
            'dp': dp_compare(original.get_reading_order(), ro).recall()
        }

    prima = [results[file][i]['prima'] for i in results[file]]
    dp = [results[file][i]['dp'] for i in results[file]]
    file_methods = [i for i in results[file]]

    barWidth = 0.25
    r1 = range(len(file_methods))
    r2 = [x + barWidth for x in r1]

    plt.figure(figsize=(10, 6))

    plt.bar(r1, prima, color='#1e81b0', width=barWidth, edgecolor='white', label='var1')
    plt.bar(r2, dp, color='#e28743', width=barWidth, edgecolor='white', label='var2')
    plt.xticks([r + barWidth for r in r1], file_methods, rotation=30)

    for a, b in zip(r1, prima):
        plt.text(a - barWidth * 2, b, str(b))

    for a, b in zip(r1, dp):
        plt.text(a + barWidth / 2, b, str(b))

    plt.title(file)
    plt.legend(['prima', 'recall'], loc='upper left')
    plt.subplots_adjust(bottom=0.18)
    plt.ylim(0, 110)
    plt.savefig(os.path.join(dir, base_filename + '.pdf'))
    plt.close()

if len(results) > 1:
    plt.close()

    total = {
        'prima': [],
        'dp': [],
        'methods': methods
    }

    collect = Collection.make(results)

    for method in methods:
        method_results = Collection.make(collect.pluck(method))
        total['prima'].append(round(Collection.make(method_results.pluck('prima')).avg(), 2))
        total['dp'].append(round(Collection.make(method_results.pluck('dp')).avg(), 2))

    barWidth = 0.25
    r1 = range(len(total['methods']))
    r2 = [x + barWidth for x in r1]

    plt.figure(figsize=(10, 5))

    plt.bar(r1, total['prima'], color='#1e81b0', width=barWidth, edgecolor='white', label='var1')
    plt.bar(r2, total['dp'], color='#e28743', width=barWidth, edgecolor='white', label='var2')
    plt.xticks([r + barWidth for r in r1], total['methods'], rotation=30)

    for a, b in zip(r1, total['prima']):
        plt.text(a - barWidth * 2, b, str(b))

    for a, b in zip(r1, total['dp']):
        plt.text(a + barWidth / 2, b, str(b))

    # plt.title('Total avg')
    plt.legend(['Prima', 'Recall'], loc='upper left')
    plt.subplots_adjust(bottom=0.18)
    plt.ylim(0, 100)
    # plt.show()
    plt.savefig(os.path.join(dir, 'total.pdf'))
    plt.clf()

    plt.plot(r1, total['prima'], color='#1e81b0', label='var1', marker='.')
    plt.plot(r2, total['dp'], color='#e28743', label='var2', marker='.')

    methods = [m.replace('smooth-', '') for m in methods]
    plt.xticks([r + barWidth for r in r1], methods)

    for a, b in zip(r1, total['prima']):
        plt.text(a - barWidth * 2, b, str(b))

    for a, b in zip(r1, total['dp']):
        plt.text(a + barWidth / 2, b, str(b))

    plt.title('Total avg')
    plt.legend(['Prima', 'Recall'])
    plt.subplots_adjust(bottom=0.18)
    # plt.savefig(os.path.join(dir, 'total-line.pdf'), format="pdf", bbox_inches='tight', pad_inches=0)

