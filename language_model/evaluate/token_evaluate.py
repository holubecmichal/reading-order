import torch

from .base import Base


class TokenEvaluate(Base):
    def _get_candidate_tokens(self, position, read_tokens_count, candidate_tokens_count):
        candidates = []

        for i in range(0, len(self.offsets)):
            offset_val = self.offsets[i]

            if offset_val == -1:
                offset_val = torch.randint(0, len(self.content), (1,)).item()

            offset = position + offset_val + read_tokens_count
            tokens = self.content_token.get(offset, candidate_tokens_count)
            candidates.append(tokens)

        return candidates

    def _read(self, position, read_count):
        read_tokens = self.content_token.get(position, read_count)
        return self.model.read_tokens(read_tokens)

    def _get_pieces_count(self):
        max_read = max(self.read_lengths)
        max_candidate_length = max(self.candidate_lengths)
        max_offset = max(self.offsets)

        max_need_length = max_read + max_candidate_length + max_offset

        return (len(self.content_token) - max_need_length) // self.samples
