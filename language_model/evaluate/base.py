import time
from abc import abstractmethod, ABC

from e_results import EResults
from language_model.carrier import carrier


class Base(ABC):
    def __init__(self, device, samples, read_lengths, candidate_lengths, offsets):
        self.device = device
        self.candidate_lengths = candidate_lengths
        self.read_lengths = read_lengths
        self.samples = samples
        self.offsets = offsets
        self.candidates_count = len(offsets)
        (self.content, self.content_token, self.model, self.vocab) = carrier(device)
        self.start_time = None

    def eval(self, start_position=None, stop_position=None):
        results = self._init_results()
        piece_size = self._get_pieces_count()

        if start_position is None and stop_position is None:
            start_position = 0
            stop_position = self.samples

        for i in range(start_position, stop_position):
            self._start(i)
            position = piece_size * i

            for read_count in self.read_lengths:
                read_output, read_hidden = self._read(position, read_count)

                for candidate_count in self.candidate_lengths:
                    candidate_tokens = self._get_candidate_tokens(position, read_count, candidate_count)

                    probs = self.model.estimate(candidate_tokens, read_output, read_hidden)
                    results.add_result(i, probs, position, read_count, candidate_count)

            self._stop()

        return results

    def _init_results(self):
        return EResults(self.samples, self.read_lengths, self.candidate_lengths, self.offsets)

    def _start(self, i):
        self.start_time = time.time()
        print('sample: ' + str(i + 1))

    def _stop(self):
        print('| time: {:5.2f}s |'.format(time.time() - self.start_time))

    @abstractmethod
    def _read(self, position, read_count):
        pass

    @abstractmethod
    def _get_candidate_tokens(self, position, read_count, candidate_count):
        pass

    @abstractmethod
    def _get_pieces_count(self):
        pass
