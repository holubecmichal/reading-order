import os.path
import pickle

from utils.list import flatten
from .constants import CS_DATA, get_data_path
from .vocabulary import Vocabulary
from nltk import tokenize

SHIFT_LEFT = 'left'
SHIFT_RIGHT = 'right'

class Content(object):
    """
    Trida pro uchovani textoveho obsahu korpusu
    """

    def __init__(self):
        self._word2idx = {}
        self._idx2word = []
        self.content = []

    def _add_word(self, word):
        if word not in self._word2idx:
            self._idx2word.append(word)
            self._word2idx[word] = len(self._idx2word) - 1
        return self._word2idx[word]

    def add(self, content):
        words = content.split()
        for word in words:
            id = self._add_word(word)
            self.content.append(id)

    def get(self, position, words_count):
        """
        Metoda, ktera vraci text o danem mnozstvi slov a dle pozadovane pozice a
        """

        ids = []
        content_length = len(self)

        for i in range(position, position + words_count):
            id = self.content[i % content_length]
            word = self._idx2word[id]
            ids.append(word)

        return ' '.join(ids)

    def __len__(self):
        return len(self.content)


def load_content(filename: str) -> Content:
    """
    Metoda pro nacteni instance Content obsahujici textovy obsah korpusu
    """

    file = os.path.join(CS_DATA, filename)
    binary = file + '.pkl'

    # pokud existuje prezpracovana binarni podoba, nactu ji a vratim
    if os.path.isfile(binary):
        with open(binary, 'rb') as f:
            return pickle.load(f)

    # jinak zpracuji a ulozim binarni metodu pro pripadne budouci pouziti
    content = Content()
    with open(file, 'r', encoding="utf8") as f:
        for line in f:
            content.add(line)

    with open(binary, 'wb') as f:
        pickle.dump(content, f)

    return content


def get_pair_sentences(root: str, file: str, vocab: Vocabulary, tokenize_lang: str, pair_length):
    """
    Metoda zpracuje obsah a identifikuje vety, ktere mohou slouzit pro analyzu uspesnosti jazykoveho modelu
    v pripade, kdy Source vetu konci a Candidate vetu zacina.

    Slouzila k experimentum, jak jazykovy model odhaduje pokud Source je konec vety a Candidate vetu zacina.
    V DP nakonec nevyuzito, v kodu ponechavam k pro pripadne dalsi ucely
    """

    file = get_data_path(root, file)
    with open(file, 'r') as f:
        lines = f.readlines()

    results = []
    for line in lines:
        total_tokens = vocab.Encode(line)
        if len(total_tokens) < pair_length*2+6:
            continue

        sentences = tokenize.sent_tokenize(line, tokenize_lang)
        sentences_tokens = vocab.Encode(sentences)
        chunks = []

        length = 0
        a = None
        b = None
        for i, tokens in enumerate(sentences_tokens):
            length += len(tokens)
            chunks.append(tokens)

            if length < pair_length+1:
                continue

            if a is None:
                a = flatten(chunks)
                chunks = []
                length = 0
                continue

            if b is None:
                b = flatten(chunks)
                results.append((a, b))

                a = b
                b = None
                length = 0
                chunks = []

    return results


def filter_by_shift_length(pairs: [tuple], shift: int, pair_length: int = 64):
    return [x for x in pairs if len(x[0]) >= pair_length + shift and len(x[1]) >= pair_length + shift]


def shift_pairs(pair: tuple, shift: int, dir: str):
    return [shift_pair(pair, shift=shift, dir=dir) for pair in pair]


def shift_pair(pair: tuple, shift: int, dir: str) -> tuple:
    a, b = pair

    if dir == SHIFT_LEFT:
        tokens = a
        left = tokens[0:-shift]
        right = tokens[len(tokens) - shift:]
        return left, right + b

    if dir == SHIFT_RIGHT:
        tokens = b
        left = tokens[0:shift]
        right = tokens[shift:]
        return a + left, right
