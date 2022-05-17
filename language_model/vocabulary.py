import os.path

import sentencepiece as spm
import torch

from .constants import get_vocab_path, get_data_path


class Vocabulary(spm.SentencePieceProcessor):
    """
    Trida dedi SentencePiece processor a rozsiruje ji o pomocnou metodu pro prevod textu na tokeny
    """

    def text_to_token_tensor(self, text):
        tokens = self.Encode(text)

        if isinstance(text, list):
            return list(map(torch.tensor, tokens))

        return torch.tensor(tokens)


def load_vocab(root_folder: str, source: str, size: int) -> Vocabulary:
    """
    Funkce pro nacteni slovniku.
    Pokud slovnik neexistuje, je ze Source vytvoren pomoci SentencePiece dle pozadovane velikosti Size
    """

    prefix, model, vocab = get_vocab_path(root_folder, source, size)
    input = get_data_path(root_folder, source)

    if not os.path.isfile(model):
        spm.SentencePieceTrainer.Train(input=input, model_prefix=prefix,
                                       vocab_size=size, max_sentence_length=8384)

    vocab = Vocabulary()
    vocab.LoadFromFile(model)

    return vocab