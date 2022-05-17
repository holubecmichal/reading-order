import copy
import numpy as np
import torch

from matplotlib import pyplot as plt
from e_results import EResults
from language_model.evaluate.results_graphs import heatmap
from utils.collection import Collection

"""
Skript pro vyhodnoceni ziskanych vysledky z language_model_evaluate.py
a jejich vizualizace
"""

path = 'cs/results/'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

results_cache = {}

def get_matrix_result(eresults, op='mean'):
    collection = Collection.make(eresults.results)

    # setrizeni vysledku podle delky 'cteneho textu'
    groups = collection.group_by('read_length')

    # cache pro pripadne znovupouziti
    oid = ''.join([str(id(eresults)), op])
    if oid in results_cache:
        return results_cache[oid]

    results = []
    for read_length in groups:
        read_samples = groups[read_length]
        row = []

        # T - delka kandidata, kterou budu vyhodnocovat
        for T in eresults.read_lengths:
            correct = 0

            for sample in read_samples:
                # P(kandidat)
                probs = sample.get_probs()

                if op == 'mean':
                    # P(kandidat[:T]|prefix) - P(kandidat[:T])
                    aggregated = [cond_probs[:T].mean() - probs[i][:T].mean() for i, cond_probs in enumerate(sample.get_cond_probs())]
                elif op == 'sum':
                    aggregated = [cond_probs[:T].sum() - probs[i][:T].sum() for i, cond_probs in enumerate(sample.get_cond_probs())]
                else:
                    raise Exception('Unknown operation')

                key = aggregated.index(max(aggregated))
                if key == 0:
                    correct += 1

            row.append(correct / len(read_samples) * 100)
        results.append(row)

    results_cache[oid] = results
    return results


def filter_candidates_count(eresults, count):
    ceresults = copy.deepcopy(eresults)

    for i, item in enumerate(ceresults.results):
        item.probabilities = item.probabilities[:count]
        ceresults.results[i] = item

    return ceresults


def left_strength_heatmap(eresults, op='mean', vmin=None, vmax=None, title=None, xlabel=None, ylabel=None,
                          maxannotate=False, cbar=True):

    results = get_matrix_result(eresults, op)
    return heatmap(results, xticklabels=eresults.read_lengths, yticklabels=eresults.read_lengths, title=title,
                   vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel, maxannotate=maxannotate, cbar=cbar)


def left_strength_plot(eresults, op='mean'):
    fig, ax = plt.subplots()
    results = np.array(get_matrix_result(eresults, op)).mean(axis=0)
    xi = list(range(len(eresults.read_lengths)))

    ax.plot(xi, results, linestyle='-', marker='o', label=op.upper())
    ax.set_xticks(xi, eresults.read_lengths)
    return ax

def vmin_vmax(*args):
    vmin = np.inf
    vmax = 0

    for i in args:
        res = np.array(get_matrix_result(i))
        vmin = min(vmin, res.min())
        vmax = max(vmax, res.max())

    return vmin, vmax


def print_stats(eresults):
    results = np.array(get_matrix_result(eresults))
    borders_std = np.max(results, axis=1) - np.std(results, axis=1) / 2
    borders_fst = results[:,0]
    weights = []

    for i, vals in enumerate(results):
        border_std = borders_std[i]
        border_fst = borders_fst[i]

        # border = max(border_std, border_fst)
        border = border_std
        keys = np.where(vals >= border)[0]
        left = keys[0]
        right = keys[-1]
        interval = (eresults.read_lengths[left], eresults.read_lengths[right])
        argmax = eresults.read_lengths[vals.argmax()]
        maxval = vals.max()

        ratio = vals[keys] / vals[keys].sum()
        read_weights = {}
        for j, _ in enumerate(keys):
            candidate_count = eresults.read_lengths[keys[j]]
            read_weights[candidate_count] = round(ratio[j], 3)

        weights.append((eresults.read_lengths[i], read_weights))

        print("{:>2}: max: {:<1} {:2.2f}, border_std={:<5} border_fst={:<5} win: {:<5} result: {}".format(
            eresults.read_lengths[i], argmax, maxval, round(border_std, 3), round(border_fst, 3), round(border, 3),
            interval))


def plot_compare(eresults_left, eresults_right):
    xi = list(range(len(eresults_left.read_lengths)))
    fix, ax = plt.subplots()
    ax.plot(xi, np.array(get_matrix_result(eresults_left)).mean(axis=0), linestyle='-', marker='.', color='C0',
            label='left shift')
    ax.plot(xi, np.array(get_matrix_result(eresults_right)).mean(axis=0), linestyle='-', marker='.', color='C1',
            label='right shift')
    ax.legend()

    eresults_left8 = filter_candidates_count(eresults_left, 8)
    eresults_right8 = filter_candidates_count(eresults_right, 8)

    ax.plot(xi, np.array(get_matrix_result(eresults_left8)).mean(axis=0), linestyle='--', marker='.', color='C0',
            label='8 left shift')
    ax.plot(xi, np.array(get_matrix_result(eresults_right8)).mean(axis=0), linestyle='--', marker='.', color='C1',
            label='8 right shift')

    eresults_left4 = filter_candidates_count(eresults_left, 4)
    eresults_right4 = filter_candidates_count(eresults_right, 4)

    ax.plot(xi, np.array(get_matrix_result(eresults_left4)).mean(axis=0), linestyle='-.', marker='.', color='C0',
            label='4 left shift')
    ax.plot(xi, np.array(get_matrix_result(eresults_right4)).mean(axis=0), linestyle='-.', marker='.', color='C1',
            label='4 right shift')

    eresults_left2 = filter_candidates_count(eresults_left, 2)
    eresults_right2 = filter_candidates_count(eresults_right, 2)

    ax.plot(xi, np.array(get_matrix_result(eresults_left2)).mean(axis=0), linestyle=':', marker='.', color='C0',
            label='2 left shift')
    ax.plot(xi, np.array(get_matrix_result(eresults_right2)).mean(axis=0), linestyle=':', marker='.', color='C1',
            label='2 right shift')

    ax.set_xticks(xi, eresults_left.read_lengths)
    return ax


np.set_printoptions(linewidth=np.inf)
eresults = EResults.load(path, "random_token_16_20000_10000.pkl")
eresults8 = filter_candidates_count(eresults, 8)
eresults4 = filter_candidates_count(eresults, 4)
eresults2 = filter_candidates_count(eresults, 2)
# vmin, vmax = vmin_vmax(eresults, eresults8)
vmin, vmax = 16, 90

print(16)
left_strength_heatmap(eresults, ylabel='Počet tokenů inicializační sekvence', xlabel='Počet tokenů kandidátních sekvencí', maxannotate=False, vmax=vmax, vmin=vmin)
plt.show()

print(8)
res = np.array(get_matrix_result(eresults8))
print(res)
print('min {}, max {}, mean {}'.format(res.min(), res.max(), res.mean()))
left_strength_heatmap(eresults8, ylabel='Počet tokenů inicializační sekvence', xlabel='Počet tokenů kandidátních sekvencí', maxannotate=False, vmax=vmax, vmin=vmin)
plt.show()

print(4)
left_strength_heatmap(eresults4, ylabel='Počet tokenů inicializační sekvence', xlabel='Počet tokenů kandidátních sekvencí', maxannotate=True, vmax=vmax, vmin=vmin)
plt.show()

print(2)
left_strength_heatmap(eresults2, ylabel='Počet tokenů inicializační sekvence', xlabel='Počet tokenů kandidátních sekvencí', maxannotate=True, vmax=vmax, vmin=vmin)
plt.show()


print("MEAN 16")
print(np.array(get_matrix_result(eresults)))
print_stats(eresults)
print("MEAN 8")
print(np.array(get_matrix_result(eresults8)))
print_stats(eresults8)
print("MEAN 4")
print(np.array(get_matrix_result(eresults4)))
print_stats(eresults4)
print("MEAN 2")
print(np.array(get_matrix_result(eresults2)))
print_stats(eresults2)

xi = list(range(len(eresults.read_lengths)))
fig, ax = plt.subplots()
ax.plot(xi, np.array(get_matrix_result(eresults)).mean(axis=0), linestyle='-', marker='.', label='16')
ax.plot(xi, np.array(get_matrix_result(eresults8)).mean(axis=0), linestyle='-', marker='.', label='8')
ax.plot(xi, np.array(get_matrix_result(eresults4)).mean(axis=0), linestyle='-', marker='.', label='4')
ax.plot(xi, np.array(get_matrix_result(eresults2)).mean(axis=0), linestyle='-', marker='.', label='2')
ax.set_xticks(xi, eresults.read_lengths)
ax.legend()
ax.set_ylim(10, 100)
plt.show()