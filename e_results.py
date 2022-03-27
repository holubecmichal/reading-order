from __future__ import annotations

import os
import pickle
from typing import Iterator

import torch


class EResults(object):
    def __init__(self, samples, read_lengths, candidate_lengths, candidate_offsets):
        self.samples = samples
        self.read_lengths = read_lengths
        self.candidate_lengths = candidate_lengths
        self.candidate_offsets = candidate_offsets
        self.results = []

    @staticmethod
    def load(path, binary):
        print('Loading results from pickle ...')
        with open(os.path.join(path, binary), 'rb') as f:
            obj = pickle.load(f)
            return obj

    def add_result(self, sample, probabilities, position, read_length, candidate_length):
        probabilities = [prob.to('cpu') for prob in probabilities]
        result = EResultItem(self, sample, probabilities, position, read_length, candidate_length)
        self.results.append(result)

        return result

    def get_candidate_count(self) -> int:
        return len(self.candidate_offsets)

    def save(self, name):
        print('Save results ...')

        # stahnout data na cpu
        for result in self.results:
            result.probabilities = [prob.to('cpu') for prob in result.probabilities]

        with open(name, 'wb') as f:
            pickle.dump(self, f)

    def print_result(self):
        collection = self.get_results()

        for read_length in self.read_lengths:
            read_collection = collection.get_by_read_length(read_length)

            for candidate_length in self.candidate_lengths:
                results = read_collection.get_by_candidate_length(candidate_length)
                sum_histogram = results.get_sum_histogram()
                avg_histogram = results.get_avg_histogram()

                print('| read_length: {} | candidate_length: {} |  {:>4} | '.format(read_length, candidate_length, ' '.join(map(str, sum_histogram))))
                print('| read_length: {} | candidate_length: {} |  {:>4} | '.format(read_length, candidate_length, ' '.join(map(str, avg_histogram))))

        print('| {:>4} | '.format(' '.join(map(str, self.get_results().get_sum_histogram()))))
        print('| {:>4} | '.format(' '.join(map(str, self.get_results().get_avg_histogram()))))

    def get_results(self) -> EResultCollection:
        return EResultCollection(self.results, self)


class EResultItem(object):
    def __init__(self, eresult, sample, probabilities, position, read_length, candidate_length):
        self.eresult = eresult
        self.candidate_length = candidate_length
        self.read_length = read_length
        self.position = position
        self.probabilities = probabilities
        self.sample = sample

    def get_sum_probabilities(self):
        probs = [self.probabilities[i].sum() for i in range(0, len(self.probabilities))]
        probs = torch.tensor(probs)
        return probs

    def get_avg_probabilities(self):
        probs = [self.probabilities[i].mean() for i in range(0, len(self.probabilities))]
        probs = torch.tensor(probs)
        return probs


class EResultCollection(object):
    def __init__(self, results: list[EResultItem], eresults: EResults):
        self.results = results
        self.eresults = eresults

    def get_by_read_length(self, length: int):
        return self._create([result for result in self.results if result.read_length == length])

    def get_by_candidate_length(self, length: int):
        return self._create([result for result in self.results if result.candidate_length == length])

    def get_sum_winners(self) -> list[int]:
        return [result.get_sum_probabilities().argmax().item() for result in self.results]

    def get_avg_winners(self) -> list[int]:
        return [result.get_avg_probabilities().argmax().item() for result in self.results]

    def get_sum_histogram(self) -> list[int]:
        sumprobs = self.get_sum_winners()
        return [sumprobs.count(key) for key in range(0, self.eresults.get_candidate_count())]

    def get_avg_histogram(self) -> list[int]:
        avgprobs = self.get_avg_winners()
        return [avgprobs.count(key) for key in range(0, self.eresults.get_candidate_count())]

    def _create(self, data):
        return EResultCollection(data, self.eresults)

    def __iter__(self) -> Iterator[EResultItem]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)
