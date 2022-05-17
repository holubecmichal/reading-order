import re
from copy import copy

import torch
import scipy.special
from torch import Tensor

from document.stubs import TextRegion as StubTextRegion, Document as StubDocument
from language_model.model import Model
from reading_order.reading_order import ReadingOrder, Group
from language_model.vocabulary import Vocabulary

TOKEN_LIMIT = 64

SETTINGS = {
    2: {
        1: 2,
    },
    4: {
        1: 2,
        40: 3,
    },
    8: {
        1: 2,
        32: 3,
    },
    16: {
        1: 2,
        12: 3,
    }
}

SETTINGS_SENTENCE = {
    2: {
        1: 1,
        2: 4,
        12: 12,
        48: 14,
    },
    4: {
        1: 1,
        7: 3,
        10: 12,
        56: 14,
    },
    8: {
        1: 1,
        8: 5,
        16: 14,
    },
    16: {
        1: 1,
        14: 14,
    },
}


def is_end_of_sentence(candidate: StubTextRegion):
    candidate = candidate.get_text().strip()

    if len(candidate) < 3:
        return False

    if candidate[-1] not in ['.', '!', '?']:
        return False

    match = re.search('[^\d](\d{1,2}).$', candidate)
    return not match


def unravel_index(weights, candidates):
    """
    Metoda vrati indexy pro source a successor dle nejvetsi hodnoty pravdepodobnosti
    """

    # beru prvek s nejvetsi pravdepodobnosti
    max_index = weights.argmax().item()

    # na zaklade indexu matice zjistim id zdroje (source) a naslednika (successor)
    index_source = max_index // len(candidates)
    index_successor = max_index % len(candidates)

    keys = list(candidates.keys())
    key_source = keys[index_source]
    key_successor = keys[index_successor]

    return key_source, key_successor


class AnalyzeSettings(object):
    def __init__(self):
        self.hard_limit = False
        self.analyze_sentences = False
        self.score = False


class LmAnalyzer(object):
    def __init__(self, model: Model, vocab: Vocabulary):
        self.model = model
        self.vocab = vocab
        self.settings = AnalyzeSettings()
        # defaultni nastaveni
        self.use_score_hard_limit(5)

    def analyze(self, doc: StubDocument) -> ReadingOrder:
        """
        Metoda pro jazzkovou analyzu dokumentu.
        Provede odhad pravdepodobnosti pro vsechny textove regiony a postupne vytvari reading order
        dle nejvyssi pravdepodobnosti
        """

        candidates = copy(doc.get_text_regions())
        candidates = {x: candidates[x] for x in candidates if len(candidates[x].get_text())}
        return Processor(candidates, self.model, self.vocab, self.settings).analyze()

    def analyze_one(self, source: StubTextRegion, candidates: {StubTextRegion}):
        """
        Metoda pro kombinovanou analyzu, kdy je pro jeden Source element vyhodnocena pravdepodobnost
        vsech Candidates.
        """

        return Processor(candidates, self.model, self.vocab, self.settings).analyze_one(source)

    def set_soft_limit(self, sentences=False):
        self.settings.score = False
        self.settings.hard_limit = None
        self.settings.analyze_sentences = sentences

    def use_hard_limit(self, tokens: int):
        self.settings.score = False
        self.settings.hard_limit = tokens
        self.settings.analyze_sentences = False

    def use_score_hard_limit(self, tokens: int = None):
        self.settings.score = True

        if tokens:
            self.settings.hard_limit = tokens
        else:
            self.settings.hard_limit = None

        self.settings.analyze_sentences = False


class Processor():
    def __init__(self, candidates: {StubTextRegion}, model: Model, vocab: Vocabulary, settings: AnalyzeSettings):
        """
        candidates - textove regiony pro jazykovou analyzu
        model - jazykovy model, se kterym je analyza provedena
        vocab - instance SentencePiece, pomoci ktere je prevedena textova sekvence na sekvenci tokenu
        settings - nastaveni analyzy
        """

        self.vocab = vocab
        self.model = model
        self.candidates = candidates
        self.settings = settings

        # nasteni tokenu pro vsechny textove sekvence
        tokens = self.vocab.Encode([self.candidates[x].get_text() for x in self.candidates])
        tokens = [torch.tensor(x) for x in tokens]
        self.tokens = self._to_dict(self.candidates.keys(), tokens)

        self.hidden_layers = {}
        self.results = {}
        self.end_of_sentences = {}
        self._init_end_of_sentences()

        if self.settings.score:
            # v pripade score, prednacteni pravdepodobnosti kandidatu
            est_tokens = [self.tokens[i][0:self._get_limit(i)] for i in self.tokens]

            read = self.model.read_init()
            self.probs = self.model.estimate(est_tokens, read)

    def _to_dict(self, keys, values):
        return dict(zip(keys, values))

    def _dict_to_list(self, dictionary):
        return list(dictionary.values())

    def _candidates_count(self):
        return len(self.candidates)

    def _init_hidden(self):
        """
        Metoda provede inicializaci skrytych vrstev pro kazdeho kandidata
        """

        print('init hidden')

        for i in self.candidates:
            source = self.candidates[i]
            # 'precteni' kandidata jayzkovym modelem
            self.hidden_layers[i] = self._read(source)

    def _is_end_of_sentence(self, source):
        return is_end_of_sentence(source) if self.settings.analyze_sentences else False

    def _init_end_of_sentences(self):
        """
        Predzpracovani pro zjisteni, ktery kandidat konci vetou
        """

        for i in self.candidates:
            source = self.candidates[i]
            self.end_of_sentences[i] = self._is_end_of_sentence(source)

    def _join(self, tensor: Tensor, ordered_group: Group):
        """
        Spojeni kandidatu, pro ktere je odhadhuta nejvetsi pravdepodobnost
        """

        # pro kazdeho kandidata softmax - zjistim pravdepodobnosti
        weights = scipy.special.softmax(tensor)
        key_source, key_successor = unravel_index(weights, self.candidates)
        source = self.candidates[key_source]
        successor = self.candidates[key_successor]

        # pridani zdroje a naslednika do reading order
        ordered_group.add_candidates(source, successor)

        # odstraneni puvodnich dat
        self._remove_joined(key_source, key_successor)

        item = ordered_group.items[source.get_id()]
        id = item.get_first().get_id()

        # nahrazeni novym (sloucenym) prvkem
        self.candidates[id] = item
        self.hidden_layers[id] = self._read(item)
        self.tokens[id] = torch.tensor(self.vocab.Encode(item.get_text()))
        self.end_of_sentences[id] = self._is_end_of_sentence(item)

        return id

    def analyze_one(self, source: StubTextRegion):
        """
        Analyza pravdepodobnosti Candidates pro predany Source
        Metoda pro Kombinovanou analyzu, zde neni potreba pocitat metodou kazdy s kazdym
        """

        id = source.get_id()

        # inicializace skrytych stavu dle Source
        self.hidden_layers[id] = self._read(source)
        self.end_of_sentences[id] = self._is_end_of_sentence(source)
        self.tokens[id] = torch.tensor(self.vocab.Encode(source.get_text()))

        if self.settings.score:
            self.probs.append(torch.tensor((.0,)))

        # odhad pravdepodobnosti
        results = self._estimate(source)
        del results[id]

        results = list(results.values())
        results = [i.to('cpu') for i in results]

        # softmax a vraceni slovniku {id: prob}, kde id je identifikator kandidata
        # a prob je pravdepodobnost sekvence Source a Candidate
        probs = scipy.special.softmax(results)
        return self._to_dict(self.candidates.keys(), probs)

    def analyze(self):
        """
        Jayzkova analyza, porovnani kazdy s kazdym a postupna tvorba Reading Order dle nejvyssi
        pravdepodobnosti kandidatu
        """

        # inicializace ReadingOrder
        reading_order = ReadingOrder()
        ordered_group = reading_order.root.add_ordered_group()

        # inicializace skrytych stavu
        self._init_hidden()

        # vypis zpracovani
        print(self._candidates_count())

        # matice kazdy s kazdym
        tensor = self._calculate()
        # spojeni dvou kandidatu dle nejvyssi pravdepodobnost, processed_id - id noveho spojeneho prvku
        processed_id = self._join(tensor, ordered_group)

        # dokud nejsou vsichni kandidati spojeni, procesuju, odhaduju a spojuju
        while len(self.candidates) > 1:
            print(self._candidates_count())
            # zpracovani noveho, spojeneho prvku, inicializace jeho skrytych stavu
            tensor = self._calculate_processed(processed_id)
            # spojeni dvou kandidatu dle nejvyssi pravdepodobnost, processed_id - id noveho spojeneho prvku
            processed_id = self._join(tensor, ordered_group)

        return reading_order

    def _read(self, source: StubTextRegion):
        # inicializace skryte vrstvy pro Source
        return self.model.read_text(source.get_text(), self.vocab)

    def _estimate(self, source: StubTextRegion, candidate: StubTextRegion = None):
        """
        Odhad pravdepodobnosti.
        Source - inicializacni sekvence
        Canididates - kanidati
        """
        limit = self._get_limit(source.get_id())

        # zpracovani jen urciteho mnozstvi tokenu
        if candidate:
            tokens = [self.tokens[candidate.get_id()][0:limit]]
        else:
            tokens = [self.tokens[x][0:limit] for x in self.tokens]

        # nacteni skryte vrstvy pro source a odhad pravdepodobnosti
        hidden = self.hidden_layers[source.get_id()]
        probs = self.model.estimate(tokens, hidden)

        if self.settings.score:
            probs = [x.mean() - self.probs[i][0:limit].mean() for i, x in enumerate(probs)]
        else:
            probs = [x.mean() for x in probs]

        if candidate:
            return probs
        else:
            return self._to_dict(self.tokens.keys(), probs)

    def _dict_to_tensor(self, dict):
        tensor = torch.empty((0, len(dict)))

        for row in dict:
            probs = torch.tensor([dict[row][x] for x in dict[row]])
            tensor = torch.vstack((tensor, probs))

        tensor.fill_diagonal_(tensor.min() - 1)
        return tensor

    def _calculate(self) -> torch.Tensor:
        """
        Odhad pravepodobnosti vsech kandidatu metodu kazdy s kazdym
        """

        for i in self.candidates:
            self.results[i] = self._estimate(self.candidates[i])

        return self._dict_to_tensor(self.results)

    def _calculate_processed(self, id):
        """
        Zpracovani noveho prvku, zpracovani radku a sloupce tohoto prvku
        """

        candidate = self.candidates[id]

        # nacteni sloupce
        for i in self.results:
            source = self.candidates[i]
            self.results[i][id] = self._estimate(source, candidate)[0]

        #nacteni radku
        source = self.candidates[id]
        self.results[id] = self._estimate(source)

        return self._dict_to_tensor(self.results)

    def _get_limit(self, source_id):
        """
        Pro pro dany textovy region vraci pocet tokenu, ktere se maji zpracovat
        """

        if self.settings.hard_limit:
            return self.settings.hard_limit

        last = None

        if self.end_of_sentences[source_id]:
            original = SETTINGS_SENTENCE
        else:
            original = SETTINGS

        settings = SETTINGS[max(SETTINGS)]
        for i in original:
            if i > len(self.candidates):
                break

            settings = original[i]

        for i in settings:
            if i > len(self.tokens[source_id]):
                break

            last = i

        return settings[last]

    def _remove_joined(self, key_source, key_successor):
        """
        Vycisteni od prvku, ktere byly spojeny do noveho elementu
        """

        # odstraneni jejich sloupcu
        for i in self.results:
            del self.results[i][key_source]
            del self.results[i][key_successor]

        # odstraneni textovych sekvenci
        del self.candidates[key_source]
        del self.candidates[key_successor]

        # odstraneni tokenu
        del self.tokens[key_source]
        del self.tokens[key_successor]

        # odstraneni radku
        del self.results[key_source]
        del self.results[key_successor]

        # odstraneni skrytych stavu
        del self.hidden_layers[key_source]
        del self.hidden_layers[key_successor]
