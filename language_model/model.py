import torch
import torch.nn.utils.rnn as rnn_utils

import model as m
from .constants import CS, get_model_path
from .vocabulary import Vocabulary


class Model(m.RNNModel):
    def read_tokens(self, tokens):
        hidden = self.init_hidden(1)
        tokens = tokens.view(1, -1).t()

        with torch.no_grad():  # no tracking history
            # vyhodnoceni vstupni sekvence
            return self(tokens, hidden)

    def read_text(self, text: str, vocab: Vocabulary):
        tokens = vocab.text_to_token_tensor(text).to(self._get_device())
        return self.read_tokens(tokens)

    def estimate(self, candidates_tokens, read_output, read_hidden):
        candidates_count = len(candidates_tokens)

        # zarovnani tokenu na stejnou delku
        padding_tokens = self._padding_tokens(candidates_tokens)
        # rozkopirovani skryte vrstvy pro kazdeho kandidata
        candidate_hidden = self._prepare_hidden_for_candidates(read_hidden, candidates_count)
        # protahnuti tokenu modelem
        with torch.no_grad():  # no tracking history
            candidate_outputs, _ = self(padding_tokens, candidate_hidden, False)

        # Vypocteme pocet tokenu jednotlivych kandidatu.
        # Dulezite predevsim pri zpracovani poctu slov, kdy kazde slovo muze byt slozeno z vice tokenu
        # a tedy i kazdy kandidat muze mit jiny pocet tokenu
        candidates_token_count = [len(x) for x in candidates_tokens]

        outputs = []
        # Priprava vystupnich pravdepodobnosti pro kazdeho kandidata.
        # Kvuli paddingu je nutne si vytahnout jen tolik hodnot, kolik odpovida puvodnimu poctu tokenu na daneho kandidata
        for candidate_index, tokens_count_of_candidate in enumerate(candidates_token_count):
            # Posledni prvek skryte vrstvy cteneho textu predstavuje pravdepodobnost prvniho tokenu kandidata
            prob_of_first_token = read_output[-1:]
            # Pro daneho kandidata nacteme pravdepodobnosti zbylych tokenu, krome posledniho (ten nepotrebujeme).
            # Je dulezite nacitat 0:tokens_count_of_candidate, kde tokens_count_of_candidate predstavuje pocet tokenu
            # kandidata pred zarovnanim (padding).
            prob_of_next_tokens = candidate_outputs[:, candidate_index][0:tokens_count_of_candidate - 1]
            outputs.append(torch.cat((prob_of_first_token, prob_of_next_tokens)))

        # Vytahnuti pravdepodobnosti
        probabilities = []
        for i in range(0, candidates_count):
            indices = range(0, candidates_token_count[i])
            probs = outputs[i][indices, candidates_tokens[i]]
            probabilities.append(probs)

        return probabilities

    def _padding_tokens(self, candidate_tokens):
        # zarovname tokeny na stejnou velikost - hodnota paddingu 0, zarovnani na delku nejdelsi sekvence tokenu
        return rnn_utils.pad_sequence(candidate_tokens).to(self._get_device())

    def _prepare_hidden_for_candidates(self, hidden, candidate_count):
        # naklonujeme skrytou vrstvu a vrstvu s hodnotami bunek site
        h = torch.repeat_interleave(hidden[0], candidate_count, dim=1)
        c = torch.repeat_interleave(hidden[1], candidate_count, dim=1)

        return h, c

    def _get_device(self):
        return next(self.parameters()).device


def load_model(path, name, device) -> Model:
    filepath = get_model_path(path, name)

    with open(filepath, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device(device))
        model = Model(checkpoint['model'], checkpoint['ntoken'], checkpoint['emsize'], checkpoint['nhid'],
                           checkpoint['nlayers'], checkpoint['dropout'], checkpoint['tied'])

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
