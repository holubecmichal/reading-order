import time
from abc import abstractmethod, ABC

import torch

from e_results import EResults
from language_model.carrier import carrier

STRETCH_RIGHT = 'right'
STRETCH_LEFT = 'left'

class Base(ABC):
    """
    Base trida pro vyhodnoceni chovani jazykoveho modelu,
    definuje zakladni metody a nektere i implementuje

    Je ocekavano, ze tato base trida bude podedena a implementovany prislusne metody
    """


    def __init__(self, device, samples, read_lengths, candidate_lengths, offsets):
        self.device = device
        self.candidate_lengths = candidate_lengths
        self.read_lengths = read_lengths
        self.samples = samples
        self.offsets = offsets
        self.candidates_count = len(offsets)
        self.content, self.content_token, self.model, self.vocab = carrier(device)
        self.start_time = None

        # pokud se pracuje s nahodnymi pozicemi, pak tento parametr urcuje, jestli se nahodne generovane hodnoty
        # maji zapamatovat - pokud ano, pak pri jednotlivych iteracich budou predkladany
        # stejne pozice pro nahodne kandidaty
        self.keep_rand_positions = False
        self._rand_positions = {}
        self._probs = {}
        self._start_hidden = None

        # definuje, jakym smerem se bude rozsirovat cteny text
        #
        # doleva - STRETCH_LEFT:
        # 1: 2016
        # 2: prosince 2016
        # 3: 18. prosince 2016
        # 4: dne 18. prosince 2016
        # ........
        # doprava - STRETCH_RIGHT:
        # 1: dne
        # 2: dne 18.
        # 3: dne 18. prosince
        # 4: dne 18. prosince 2016
        self.read_stretch = STRETCH_RIGHT

    def eval(self, start_position=None, stop_position=None):
        results = self._init_results()
        piece_size = self._get_pieces_count()

        if start_position is None and stop_position is None:
            start_position = 0
            stop_position = self.samples

        for i in range(start_position, stop_position):
            self._start(i)
            # pro dalsi kolo se promaze rand_positions, neni nutne uchovavat nahodne hodnoty generovane
            # pro predchozi kolo, zbytecne to zere pamet
            self._rand_positions = {}
            self._probs = {}
            position = piece_size * i

            if self.read_stretch == STRETCH_LEFT:
                # pokud se bude cteny rozsirovat doleva, je potreba pocatek posunout o maximalni delku
                position += max(self.read_lengths)

            for read_length in self.read_lengths:
                # sit si 'precte' vstupni sekvenci, inicializuje skryte stavy, ktere pouzije pro odhad
                # pravdepodobnosti navazne sekvence - kandidata
                read = self._read(position, read_length)

                for candidate_length in self.candidate_lengths:
                    # nacteni kandidatu, respektive tokenu
                    candidate_tokens = self._get_candidate_tokens(position, read_length, candidate_length)

                    # P(kandidat[:T])
                    probs = self._get_probs(candidate_length, candidate_tokens)

                    # P(kandidat[:T]|prefix)
                    cond_probs = self.model.estimate(candidate_tokens, read)

                    # ulozeni
                    results.add_result(i, cond_probs, position, read_length, candidate_length, probs)

            self._stop()

        return results

    def _get_probs(self, candidate_length, candidate_tokens):
        if self.keep_rand_positions:
            if candidate_length not in self._probs:
                self._probs[candidate_length] = self.model.estimate(candidate_tokens, self._get_start_hidden())

            probs = self._probs[candidate_length]
        else:
            probs = self.model.estimate(candidate_tokens, self._get_start_hidden())

        return probs

    def _get_start_hidden(self):
        if self._start_hidden is None:
            # torch.tensor(1) -- <s> -- start of sequence
            self._start_hidden = self.model.read_init()

        return self._start_hidden

    def _init_results(self):
        return EResults(self.samples, self.read_lengths, self.candidate_lengths, self.offsets)

    def _start(self, i):
        self.start_time = time.time()
        print('sample: ' + str(i + 1))

    def _stop(self):
        print('| time: {:5.2f}s |'.format(time.time() - self.start_time))

    def _get_rand_candidate_position(self, read_position, candidate_index):
        if self.keep_rand_positions:
            if read_position not in self._rand_positions:
                self._rand_positions[read_position] = {}

            if candidate_index not in self._rand_positions[read_position]:
                self._rand_positions[read_position][candidate_index] = self._rand()

            return self._rand_positions[read_position][candidate_index]
        else:
            return self._rand()

    def _get_candidate_start_position(self, offset_val, position, candidate_index, read_count):
        if offset_val == -1:
            start_position = self._get_rand_candidate_position(position, candidate_index)
        else:
            if self.read_stretch == STRETCH_LEFT:
                start_position = position + offset_val
            else:
                start_position = position + offset_val + read_count

        return start_position

    def _rand(self):
        return torch.randint(0, len(self.content), (1,)).item()

    @abstractmethod
    def _read(self, position, read_count):
        pass

    @abstractmethod
    def _get_candidate_tokens(self, position, read_count, candidate_count):
        pass

    @abstractmethod
    def _get_pieces_count(self):
        pass
