import os

DIR_MODELS = 'models'
DIR_VOCAB = 'vocabulary'
DIR_DATA = 'data/'

DIR = os.path.dirname(os.path.realpath(__file__))
CS = os.path.abspath(os.path.join(DIR, '../cs/'))
DE = os.path.abspath(os.path.join(DIR, '../de/'))
CS_DATA = os.path.join(CS, DIR_DATA)
CS_MODELS = os.path.join(CS, DIR_MODELS)
CS_VOCABULARY = os.path.join(CS, DIR_VOCAB)


def get_model_path(root_folder, name):
    return os.path.join(root_folder, DIR_MODELS, name)


def get_vocab_path(root_folder, source, size):
    base = os.path.join(root_folder, DIR_VOCAB, source)
    prefix = '.'.join([base, str(size)])
    model = '.'.join([prefix, 'model'])
    vocab = '.'.join([prefix, 'vocab'])
    return prefix, model, vocab


def get_data_path(root_folder, file):
    return os.path.join(root_folder, DIR_DATA, file)