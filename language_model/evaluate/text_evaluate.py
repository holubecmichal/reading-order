from .base import Base, STRETCH_LEFT


class TextEvaluate(Base):
    def _get_candidate_texts(self, position, read_length, candidate_words_count):
        """
        Metoda pro ziskani textu z dane pozice a dle pozadovane delky
        """

        candidates = []

        for i in range(0, len(self.offsets)):
            start_position = self._get_candidate_start_position(self.offsets[i], position, i, read_length)
            text = self.content.get(start_position, candidate_words_count)
            candidates.append(text)

        return candidates

    def _read(self, position, read_count):
        """
        Inicializace skrytych stavu textovym obsahem
        """

        if self.read_stretch == STRETCH_LEFT:
            real_position = position - read_count
            text = self.content.get(real_position, read_count)
        else:
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

