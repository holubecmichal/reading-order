import torch
import torch.nn.utils.rnn as rnn_utils

import model as m
from .constants import get_model_path
from .vocabulary import Vocabulary


class Model(m.RNNModel):
    """
    Trida dedi Pytorch modul LSTM site,
    poskytuje pomocne metody
    """

    def read_tokens(self, tokens):
        """
        Inicializace skrytych stavu na zaklade predanych tokenu
        """

        hidden = self.init_hidden(1)
        tokens = tokens.view(1, -1).t()

        with torch.no_grad():  # no tracking history
            # vyhodnoceni vstupni sekvence
            return self(tokens, hidden)

    def read_text(self, text: str, vocab: Vocabulary):
        """
        Inicializace skrytych stavu na zaklade predaneho textu
        """

        tokens = vocab.text_to_token_tensor(text).to(self._get_device())
        return self.read_tokens(tokens)

    def read_init(self):
        """
        Inicializace skrytych vrstev pomoci tokenu <s> -- start of sequence
        """

        return self.read_tokens(torch.tensor(1).to(self._get_device()))

    def estimate(self, tokens, hidden):
        """
        Odhad pravdepodobnosti pro kandidaty (tokens) na zaklade prefixu (hidden)
        """
        prefix_probs, hidden = hidden
        candidates_count = len(tokens)

        # rozkopirovani skryte vrstvy pro kazdeho kandidata
        hidden = self._prepare_hidden_for_candidates(hidden, candidates_count)
        # Posledni prvek skryte vrstvy cteneho textu predstavuje pravdepodobnost prvniho tokenu kandidata
        prob_of_first_token = prefix_probs[-1:]

        # zarovnani tokenu na stejnou delku
        padding_tokens = self._padding_tokens(tokens)

        # protahnuti tokenu modelem
        with torch.no_grad():  # no tracking history
            probs, _ = self(padding_tokens, hidden, False)

        # Vypocteme pocet tokenu jednotlivych kandidatu.
        # Dulezite predevsim pri zpracovani poctu slov, kdy kazde slovo muze byt slozeno z vice tokenu
        # a tedy i kazdy kandidat muze mit jiny pocet tokenu
        candidate_tokens_count = self._candidate_tokens_count(tokens)

        outputs = []
        # Priprava vystupnich pravdepodobnosti pro kazdeho kandidata.
        # Kvuli paddingu je nutne si vytahnout jen tolik hodnot, kolik odpovida puvodnimu poctu tokenu na daneho kandidata
        for i, tokens_count_of_candidate in enumerate(candidate_tokens_count):
            # Pro daneho kandidata nacteme pravdepodobnosti zbylych tokenu, krome posledniho (ten nepotrebujeme).
            # Je dulezite nacitat 0:tokens_count_of_candidate, kde tokens_count_of_candidate predstavuje pocet tokenu
            # kandidata pred zarovnanim (padding).
            prob_of_next_tokens = probs[:, i][0:tokens_count_of_candidate - 1]
            outputs.append(torch.cat((prob_of_first_token, prob_of_next_tokens)))

        # Vytahnuti pravdepodobnosti
        return self._process_probs(candidates_count, tokens, outputs)

    def _candidate_tokens_count(self, tokens):
        return [len(x) for x in tokens]

    def _process_probs(self, candidates_count, tokens, output):
        probabilities = []
        candidate_tokens_count = self._candidate_tokens_count(tokens)

        for i in range(0, candidates_count):
            indices = range(0, candidate_tokens_count[i])
            probs = output[i][indices, tokens[i]]
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
    """
    Funkce pro nacteni modelu
    """
    filepath = get_model_path(path, name)

    with open(filepath, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device(device))
        model = Model(checkpoint['model'], checkpoint['ntoken'], checkpoint['emsize'], checkpoint['nhid'],
                           checkpoint['nlayers'], checkpoint['dropout'], checkpoint['tied'])

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        return model
