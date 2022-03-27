import torch

from .base import Base


class TextEvaluate(Base):
    def _get_candidate_texts(self, position, read_length, candidate_words_count):
        candidates = []

        for i in range(0, len(self.offsets)):
            offset_val = self.offsets[i]

            if offset_val == -1:
                offset_val = torch.randint(0, len(self.content), (1,)).item()

            offset = position + offset_val + read_length
            text = self.content.get(offset, candidate_words_count)
            candidates.append(text)

        return candidates

    def _read(self, position, read_count):
        text = self.content.get(position, read_count)
        return self.model.read_text(text, self.vocab)

    def _get_candidate_tokens(self, position, words_count, candidate_words_count):
        # nacteni textu kandidatu a prevod na tokeny sentencepiece, respektive jejich tensory
        candidate_texts = self._get_candidate_texts(position, words_count, candidate_words_count)
        return self.vocab.text_to_token_tensor(candidate_texts)

    def _get_pieces_count(self):
        max_read = max(self.read_lengths)
        max_candidate_length = max(self.candidate_lengths)
        max_offset = max(self.offsets)

        max_need_length = max_read + max_candidate_length + max_offset

        return (len(self.content) - max_need_length) // self.samples

