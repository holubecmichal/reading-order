import argparse
import os
import pickle

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from utils.collection import Collection

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=str, required=True,
                    help='Results path')
parser.add_argument('--aletheia_results', type=str, required=True,
                    help='Aletheia results path')
parser.add_argument('--file_bar', action='store_true')
parser.add_argument('--avg_bar', action='store_true')

args = parser.parse_args()
results_path = os.path.abspath(args.results)
aletheia_results_path = os.path.abspath(args.aletheia_results)

if not os.path.isfile(results_path) or not os.path.isfile(aletheia_results_path):
    print('Results not found')
    exit()

with open(results_path, 'rb') as f:
    results = Collection.make(pickle.load(f))

with open(aletheia_results_path, 'rb') as f:
    aletheia_results = Collection.make(pickle.load(f))

if args.file_bar:
    df = pd.DataFrame()

    for file in aletheia_results.pluck('file').unique().sort():
        aletheia = aletheia_results.where('file', file).group_by('type')
        ros = results.where('file', file).group_by('length')
        key = file.replace('hn-12-1-2022-', '').replace('.xml', '')

        df = df.append({'type': 'ocr', 'file': key, 'penalty_percentage': aletheia['ocr'][0]['penalty_percentage']}, ignore_index=True)
        df = df.append({'type': 'top-to-bottom', 'file': key, 'penalty_percentage': aletheia['tb'][0]['penalty_percentage']}, ignore_index=True)
        df = df.append({'type': 'reading order', 'file': key, 'penalty_percentage': ros[2][0]['penalty_percentage']}, ignore_index=True)

    # Set the figure size
    plt.figure(figsize=(14, 8))

    # grouped barplot
    sns.set_theme(style="whitegrid")
    sns.barplot(x="file", y="penalty_percentage", hue="type", data=df)
    plt.title('Porovnání jednotlivých metod')
    plt.xlabel('Soubor')
    plt.ylabel('Procentuální úspěšnost')
    plt.ylim(0, 1)
    plt.legend(loc=9)
    plt.show()

if args.avg_bar:
    aletheia = aletheia_results.group_by('type')

    tb = Collection.make(aletheia['tb']).avg('penalty_percentage')
    ocr = Collection.make(aletheia['ocr']).avg('penalty_percentage')
    ro = results.avg('penalty_percentage')

    y_pos = range(3)

    # Create bars
    plt.bar(y_pos, [ocr, tb, ro])
    plt.ylim(0, 1)
    plt.title('Průměrná úspěšnost')
    plt.xlabel('Metoda')
    plt.ylabel('Procentuální úspěšnost')

    # Create names on the x-axis
    plt.xticks(y_pos, ['ocr', 'top-to-bottom', 'reading order'])
    plt.grid()

    # Show graphic
    plt.show()
