import argparse
import os.path
import pickle

from matplotlib import pyplot as plt
from utils.collection import Collection

def barplot(x, y, without_limit, title):
    y_pos = range(len(y))

    # Create bars
    plt.bar(y_pos, y)
    plt.axhline(y=without_limit, color='red')

    plt.title(title)
    plt.xlabel('Počet tokenů')
    plt.ylabel('Procentuální úspěšnost')

    # Create names on the x-axis
    plt.xticks(y_pos, x)

    # Show graphic
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=str, required=True,
                    help='Results path')
parser.add_argument('--file_bar', action='store_true')
parser.add_argument('--avg_tokens_bar', action='store_true')

args = parser.parse_args()
path = os.path.abspath(args.results)

if not os.path.isfile(path):
    print('Results not found')
    exit()

with open(path, 'rb') as f:
    results = Collection.make(pickle.load(f))

if args.file_bar:
    for file in results.pluck('file').unique().sort():
        print(file)
        file_results = results.where('file', file)

        without_limit = file_results.filter(lambda x: x['length'] is None).first()
        with_limit = file_results.filter(lambda x: x['length'] is not None)

        y = with_limit.pluck('penalty_percentage')
        x = with_limit.pluck('length')

        barplot(x=x, y=y, without_limit=without_limit['penalty_percentage'], title=file)



if args.avg_tokens_bar:
    groups = results.group_by('length')
    for i in groups:
        groups[i] = Collection.make(groups[i]).avg('penalty_percentage')

    without_limit = groups[None]
    del groups[None]

    barplot(x=groups.keys(), y=groups.values(), without_limit=without_limit, title='Průměrná úspěšnost')
