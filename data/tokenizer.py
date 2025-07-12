import re
from collections import Counter

class CaptionTokenizer:
    def __init__(self, captions, min_freq=1):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"

        tokens = [self.tokenize(c) for c in captions]
        counter = Counter(t for caption in tokens for t in caption if t)
        vocab = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        vocab += [word for word, freq in counter.items() if freq >= min_freq]

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def tokenize(self, text):
        text = re.sub(r"[^a-zA-Z0-9 ]", "", text.lower())
        return text.strip().split()

    def encode(self, text, max_len):
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(self.sos_token)]
        ids += [self.word2idx.get(t, self.word2idx[self.unk_token]) for t in tokens]
        ids.append(self.word2idx.get(self.eos_token))
        ids = ids[:max_len]
        return ids + [self.word2idx[self.pad_token]] * (max_len - len(ids))

    def vocab_size(self):
        return len(self.word2idx)
