from .constants import CS, CS_DATA, DE
from .content import Content, load_content
from .content_tokens import ContentTokens, load_content_tokens
from .model import load_model, Model
from .vocabulary import Vocabulary, load_vocab


def carrier(device) -> (Content, ContentTokens, Model, Vocabulary):
    v = default_vocab()
    c = default_content()
    ct = default_content_tokens(v, device)
    m = default_model(device)

    return c, ct, m, v


def default_vocab() -> Vocabulary:
    return load_vocab(CS, 'train.txt', 20000)


def de_vocab() -> Vocabulary:
    return load_vocab(DE, 'train.txt', 20000)


def default_content() -> Content:
    return load_content('test.txt')


def default_content_tokens(vocab: Vocabulary, device: str = 'cpu') -> ContentTokens:
    return load_content_tokens(CS_DATA, 'test.txt', vocab, device)


def default_model(device: str = 'cpu') -> Model:
    return load_model(CS, 'LSTM_vocabsize20000_emsize400_nhid1700_nlayers2_dropout0.2_batchsize20_seqlen35.tar', device)


def de_model(device: str = 'cpu') -> Model:
    return load_model(DE, 'LSTM_vocabsize20000_emsize400_nhid1700_nlayers2_dropout0.2_batchsize20_seqlen35.tar', device)
