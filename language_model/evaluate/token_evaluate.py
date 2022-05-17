from .base import Base, STRETCH_LEFT


class TokenEvaluate(Base):
    def _get_candidate_tokens(self, position, read_tokens_count, candidate_tokens_count):
        """
        Metoda pro ziskani tokenu z dane pozice a dle pozadovane delky
        """

        candidates = []

        for i in range(0, len(self.offsets)):
            start_position = self._get_candidate_start_position(self.offsets[i], position, i, read_tokens_count)
            tokens = self.content_token.get(start_position, candidate_tokens_count)
            candidates.append(tokens)

        return candidates

    def _read(self, position, read_count):
        """
        Inicializace skrytych stavu textovym obsahem
        """

        if self.read_stretch == STRETCH_LEFT:
            real_position = position - read_count
            read_tokens = self.content_token.get(real_position, read_count)
        else:
            read_tokens = self.content_token.get(position, read_count)

        return self.model.read_tokens(read_tokens)

    def _get_pieces_count(self):
        max_read = max(self.read_lengths)
        max_candidate_length = max(self.candidate_lengths)
        max_offset = max(self.offsets)

        max_need_length = max_read + max_candidate_length + max_offset

        return (len(self.content_token) - max_need_length) // self.samples
