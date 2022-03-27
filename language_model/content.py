import os.path
import pickle

from .constants import CS_DATA


class Content(object):
    def __init__(self):
        self._word2idx = {}
        self._idx2word = []
        self.content = []

    def _add_word(self, word):
        if word not in self._word2idx:
            self._idx2word.append(word)
            self._word2idx[word] = len(self._idx2word) - 1
        return self._word2idx[word]

    def add(self, content):
        words = content.split()
        for word in words:
            id = self._add_word(word)
            self.content.append(id)

    def get(self, position, words_count):
        ids = []
        content_length = len(self)

        for i in range(position, position + words_count):
            id = self.content[i % content_length]
            word = self._idx2word[id]
            ids.append(word)

        return ' '.join(ids)

    def __len__(self):
        return len(self.content)


def load_content(filename: str) -> Content:
    file = os.path.join(CS_DATA, filename)
    binary = file + '.pkl'

    if os.path.isfile(binary):
        with open(binary, 'rb') as f:
            return pickle.load(f)

    content = Content()
    with open(file, 'r', encoding="utf8") as f:
        for line in f:
            content.add(line)

    with open(binary, 'wb') as f:
        pickle.dump(content, f)

    return content
