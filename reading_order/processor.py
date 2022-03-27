from copy import copy
from typing import Optional

import torch

from language_model.model import Model
from reading_order.reading_order import ReadingOrder
from reading_order.stubs import Candidate
from language_model.vocabulary import Vocabulary
import torch.nn.functional as F


class ProcessResults(object):
    def __init__(self, results):
        self.results = results

    def mean(self) -> 'ProcessResults':
        results = {}
        for i in self.results:
            item = self.results[i]
            results[i] = dict(zip(self.results.keys(), [col.mean() for col in item.values()]))

        return ProcessResults(results)

    def tensor(self) -> torch.Tensor:
        results = []

        for i in self.results:
            results.append([self.results[i][j] for j in self.results[i]])

        tensor = torch.tensor(results)
        tensor.fill_diagonal_(tensor.min() - 1)
        return tensor


class Processor(object):
    def __init__(self, model: Model, vocab: Vocabulary):
        self.results = None
        self.processed_id = None
        self.hidden_layers = {}
        self.process_print = False

        self.model = model
        self.vocab = vocab

    def clear(self):
        self.results = None
        self.processed_id = None
        self.hidden_layers = {}

    def calculate(self, candidates: {Candidate}, token_limit: Optional[int]) -> ProcessResults:
        candidates_tokens = self._get_candidate_tokens(candidates, token_limit)

        if self.processed_id is None:
            results = {}

            for i in candidates:
                source = candidates[i]

                self.hidden_layers[i] = self.model.read_text(source.get_text(), self.vocab)
                read_output, read_hidden = self.hidden_layers[i]

                probs = self.model.estimate(list(candidates_tokens.values()), read_output, read_hidden)
                results[i] = dict(zip(candidates_tokens.keys(), probs))

            self.results = ProcessResults(results)
        else:
            source = candidates[self.processed_id]

            for i in self.results.results:
                read_output, read_hidden = self.hidden_layers[i]
                probs = self.model.estimate([candidates_tokens[self.processed_id]], read_output, read_hidden)
                self.results.results[i][self.processed_id] = probs[0]

            self.hidden_layers[self.processed_id] = self.model.read_text(source.get_text(), self.vocab)
            read_output, read_hidden = self.hidden_layers[self.processed_id]

            probs = self.model.estimate(list(candidates_tokens.values()), read_output, read_hidden)
            self.results.results[self.processed_id] = dict(zip(candidates_tokens.keys(), probs))

        return self.results

    def _get_candidate_tokens(self, candidates: {}, token_limit: Optional[int]):
        candidate_texts = {i: candidates[i].get_text() for i in candidates}
        candidates_tokens = self.vocab.text_to_token_tensor(list(candidate_texts.values()))
        candidates_tokens = dict(zip(candidates.keys(), candidates_tokens))

        if token_limit:
            for i in candidates_tokens:
                ct = candidates_tokens[i]
                candidates_tokens[i] = ct[0:token_limit]

        return candidates_tokens

    def _remove_joined(self, candidates: {Candidate}, key_source, key_successor):
        del candidates[key_source]
        del candidates[key_successor]

        del self.results.results[key_source]
        del self.results.results[key_successor]

        del self.hidden_layers[key_source]
        del self.hidden_layers[key_successor]

        for i in self.results.results:
            del self.results.results[i][key_source]
            del self.results.results[i][key_successor]

        return candidates

    def process(self, candidates: {Candidate}, token_limit=2) -> ReadingOrder:
        candidates = copy(candidates)
        reading_order = ReadingOrder()
        ordered_group = reading_order.root.add_ordered_group()

        if self.process_print:
            print(len(candidates))

        while len(candidates) > 1:
            # matice kazdy s kazdym
            results = self.calculate(candidates, token_limit)

            # vysledne spolecne pravdepodobnosti zprumerujeme (pripadne sumujeme)
            tensor = results.mean().tensor()

            # pro kazdeho kandidata softmax - zjistim pravdepodobnosti
            weights = F.softmax(tensor, dim=1)
            # beru prvek s nejvetsi pravdepodobnosti
            max_index = weights.argmax().item()

            # na zaklade indexu matice zjistim id zdroje (source) a naslednika (successor)
            index_source = max_index // len(candidates)
            index_successor = max_index % len(candidates)

            keys = list(candidates.keys())
            key_source = keys[index_source]
            key_successor = keys[index_successor]

            source = candidates[key_source]
            successor = candidates[key_successor]

            # pridani zdroje a naslednika do reading order
            ordered_group.add_candidates(source, successor)

            # odstraneni puvodnich dat a nahrazeni novym (sloucenym) prvkem
            candidates = self._remove_joined(candidates, key_source, key_successor)

            item = ordered_group.items[source.get_id()]
            candidates[item.get_first().get_id()] = item
            self.processed_id = item.get_first().get_id()

            if self.process_print:
                print(len(candidates))

        return reading_order

    def estimate(self, source: Candidate, candidates: {Candidate}, token_limit) -> dict:
        candidates_tokens = self._get_candidate_tokens(candidates, token_limit)
        read_output, read_hidden = self.model.read_text(source.get_text(), self.vocab)
        probs = self.model.estimate(list(candidates_tokens.values()), read_output, read_hidden)
        probs = F.softmax(torch.tensor([x.mean() for x in probs]), dim=0)

        results = dict(zip(candidates.keys(), probs))

        return results
