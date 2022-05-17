from __future__ import annotations

import os
import pickle

import torch

"""
Pomocne tridy pouzite pri vyhodnoceni chovani jazykoveho modelu
"""

class EResults(object):
    def __init__(self, samples, read_lengths, candidate_lengths, candidate_offsets):
        self.samples = samples
        self.read_lengths = read_lengths
        self.candidate_lengths = candidate_lengths
        self.candidate_offsets = candidate_offsets
        self.results = []

    @staticmethod
    def load(path, binary):
        """
        Nacteni serializovanych vysledku
        """

        print('Loading results from pickle ...')
        with open(os.path.join(path, binary), 'rb') as f:
            obj = pickle.load(f)
            return obj

    def add_result(self, sample, cond_probabilities, position, read_length, candidate_length, candidates_probabilities=None):
        """
        Pridani spocteneho vysledku do kolekce
        """

        cond_probabilities = [prob.to('cpu') for prob in cond_probabilities]

        if candidates_probabilities:
            candidates_probabilities = [prob.to('cpu') for prob in candidates_probabilities]

        result = EResultItem(self, sample, cond_probabilities, position, read_length, candidate_length, candidates_probabilities)
        self.results.append(result)

        return result

    def get_candidate_count(self) -> int:
        return len(self.candidate_offsets)

    def save(self, name):
        """
        Ulozeni vysledku do souboru
        """

        print('Save results ...')

        # stahnout data na cpu
        for result in self.results:
            result.probabilities = [prob.to('cpu') for prob in result.probabilities]

        with open(name, 'wb') as f:
            pickle.dump(self, f)


class EResultItem(object):
    """
    Prvek s vysledkem
    """

    def __init__(self, eresult, sample, probabilities, position, read_length, candidate_length, candidate_probabilities = None):
        self.eresult = eresult
        self.candidate_length = candidate_length
        self.read_length = read_length
        self.position = position
        self.probabilities = probabilities
        self.sample = sample
        self.candidate_probabilities = candidate_probabilities

    def has_probs(self):
        return hasattr(self, 'candidate_probabilities')

    def get_probs(self):
        if self.has_probs():
            return self.candidate_probabilities

        default = []
        for i, _ in enumerate(self.probabilities):
            default.append(torch.tensor((.0,)))

        return default


    def get_cond_probs(self):
        return self.probabilities

    def __getitem__(self, item):
        return getattr(self, item)
