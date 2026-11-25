
import re
from collections import Counter


class ScratchTokenizer:

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, vocab_size=30000, max_length=256):
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.word_to_id = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID
        }

        self.id_to_word = {
            self.PAD_ID: self.PAD_TOKEN,
            self.UNK_ID: self.UNK_TOKEN
        }

    def tokenize(self, text):
        text = text.lower()
        return re.findall(r"\b\w+\b|[^\w\s]", text)

    def fit(self, texts):

        counter = Counter()

        for text in texts:
            counter.update(self.tokenize(text))

        most_common = counter.most_common(
            self.vocab_size - 2
        )

        for index, (word, _) in enumerate(most_common, start=2):
            self.word_to_id[word] = index
            self.id_to_word[index] = word

        print(f"Vocabulary size: {len(self.word_to_id)}")

    def encode(self, text):

        tokens = self.tokenize(text)

        token_ids = [
            self.word_to_id.get(token, self.UNK_ID)
            for token in tokens
        ]

        token_ids = token_ids[:self.max_length]

        padding_length = self.max_length - len(token_ids)

        token_ids += [self.PAD_ID] * padding_length

        return token_ids

    def encode_batch(self, texts):

        import torch

        encoded = [
            self.encode(text)
            for text in texts
        ]

        return torch.tensor(
            encoded,
            dtype=torch.long
        )
