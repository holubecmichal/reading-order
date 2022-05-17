import torch

from .constants import get_data_path
from .vocabulary import Vocabulary


class ContentTokens(object):
    """
    Trida nese obsah korpusu v podobe tokenu
    """

    def __init__(self, filename: str, tokens: torch.Tensor):
        self._filename = filename
        self._tokens = tokens
        self._tokens_count = len(self._tokens)

    def __len__(self):
        return self._tokens_count

    def get(self, position, tokens_count):
        """
        Nacte a vrati sekvenci tokenu dle pozice a pozadovane delky
        """

        stop = (position + tokens_count)

        if stop > self._tokens_count:
            part1 = self._tokens[position:(self._tokens_count-1)]
            part2 = self._tokens[0:(stop-self._tokens_count+1)]
            return torch.cat((part1, part2), 0)

        return self._tokens[position:stop]

    def get_tokens(self):
        return self._tokens

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.
    def batchify(self, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = self._tokens.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = self._tokens.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data


def load_content_tokens(root_folder: str, filename: str, vocab: Vocabulary, device):
    """
    Funkce pro nacteni a prevedeni obsahu souboru na tokeny
    """

    filepath = get_data_path(root_folder, filename)

    with open(filepath, 'r', encoding="utf8") as f:
        idss = []
        for line in f:
            ids = vocab.Encode(line)

            if not len(ids):
                continue

            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)

    ids = ids.to(device)
    return ContentTokens(filename, ids)
