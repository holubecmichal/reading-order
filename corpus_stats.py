import time

from nltk.tokenize import word_tokenize

from language_model.carrier import cs_vocab
from language_model.constants import get_data_path, CS

"""
Skript pro zjisteni statistik trenovaciho korpusu
"""


def print_stats(file):
    filepath = get_data_path(CS, file)

    start_time = time.time()
    total = word_tokenize(open(filepath, 'r', encoding="utf8").read(), language='czech')
    unique = set(total)
    elapsed = time.time() - start_time
    print('File: {}, total: {}, unique: {}, elapsed: {:5.2f}s'.format(file, len(total), len(unique), elapsed))

def print_vocab_stats(file):
    filepath = get_data_path(CS, file)
    vocab = cs_vocab()

    start_time = time.time()
    total = vocab.Encode(open(filepath, 'r', encoding="utf8").read())
    unique = set(total)
    elapsed = time.time() - start_time
    print('File: {}, total: {}, unique: {}, elapsed: {:5.2f}s'.format(file, len(total), len(unique), elapsed))


print_stats('train.txt')
print_stats('valid.txt')
print_stats('test.txt')

print_vocab_stats('train.txt')
print_vocab_stats('valid.txt')
print_vocab_stats('test.txt')
