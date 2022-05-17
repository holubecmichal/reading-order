import time

import torch

from e_results import EResults
from language_model.carrier import cs_vocab, cs_model


class ShiftTokenEvaluate(object):
    """
    Trida pro experiment s posunem, respektive s koncem vety
    """


    def __init__(self, device, read_lengths, candidate_lengths, pairs, offsets):
        pairs = [(torch.tensor(a).to(device), torch.tensor(b).to(device)) for (a,b) in pairs]

        self.device = device
        self.read_lengths = read_lengths
        self.candidate_lengths = candidate_lengths
        self.offsets = offsets
        self.pairs = pairs
        self.samples = len(self.pairs)
        self.model = cs_model(self.device)
        self.vocab = cs_vocab()

        self.keep_rand_positions = False
        self._rand_positions = {}


    def eval(self):
        print("samples: {}".format(self.samples))
        results = self._init_results()

        for i in range(0, self.samples):
            self._start(i)
            self._rand_positions = {}

            for read_count in self.read_lengths:
                read = self._read(i, read_count)

                for candidate_count in self.candidate_lengths:
                    candidate_tokens = self._get_candidate_tokens(i, candidate_count)
                    cond_probs = self.model.estimate(candidate_tokens, read)
                    results.add_result(i, cond_probs, i, read_count, candidate_count)

            self._stop()

        return results

    def _read(self, i, read_count):
        a, _ = self.pairs[i]
        tokens = a[len(a)-read_count:]
        return self.model.read_tokens(tokens)

    def _get_candidate_tokens(self, pair_index, candidate_count):
        candidates = []

        for i, val in enumerate(self.offsets):
            if val == -1:
                if self.keep_rand_positions:
                    if i not in self._rand_positions:
                        self._rand_positions[i] = self._rand()
                        if pair_index == self._rand_positions[i]:
                            self._rand_positions[i] += 1

                    pair_index = self._rand_positions[i]
                else:
                    pair_index = self._rand()

                tokens = self.pairs[pair_index][1][:candidate_count]
            else:
                tokens = self.pairs[pair_index][1][val:candidate_count]

            candidates.append(tokens)

        return candidates

    def _rand(self):
        return torch.randint(0, len(self.pairs), (1,)).item()

    def _init_results(self):
        return EResults(self.samples, self.read_lengths, self.candidate_lengths, self.offsets)

    def _start(self, i):
        self.start_time = time.time()
        print('sample: ' + str(i + 1))

    def _stop(self):
        print('| time: {:5.2f}s |'.format(time.time() - self.start_time))