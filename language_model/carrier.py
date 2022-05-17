from .constants import CS, DE
from .content import Content, load_content, get_pair_sentences, filter_by_shift_length
from .content_tokens import ContentTokens, load_content_tokens
from .model import load_model, Model
from .vocabulary import Vocabulary, load_vocab

"""
Soubor obsahuje primarne pomocne metody pro nacteni modelu, obsahu trenovaciho korpusu, slovniku SentencePiece
"""

def carrier(device) -> (Content, ContentTokens, Model, Vocabulary):
    v = cs_vocab()
    c = cs_content()
    ct = cs_content_tokens(v, device)
    m = cs_model(device)

    return c, ct, m, v


def cs_vocab() -> Vocabulary:
    return load_vocab(CS, 'train.txt', 20000)


def de_vocab() -> Vocabulary:
    return load_vocab(DE, 'train.txt', 20000)


def cs_pair_sentences(pair_length=64, shift=0) -> [tuple]:
    pairs = get_pair_sentences(CS, 'test.txt', cs_vocab(), 'czech', pair_length)
    return filter_by_shift_length(pairs=pairs, shift=shift, pair_length=pair_length)


def cs_content() -> Content:
    return load_content('test.txt')


def cs_content_tokens(vocab: Vocabulary, device: str = 'cpu') -> ContentTokens:
    return load_content_tokens(CS, 'test.txt', vocab, device)


def cs_model(device='cpu') -> Model:
    return load_model(CS, 'LSTM_vocabsize20000_emsize400_nhid1700_nlayers2_dropout0.2_batchsize20_seqlen35.tar', device)


def de_model(device='cpu') -> Model:
    return load_model(DE, 'LSTM_vocabsize20000_emsize400_nhid1700_nlayers2_dropout0.2_batchsize20_seqlen35.tar', device)
