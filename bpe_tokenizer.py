# bpe_fixed.py
import json
from collections import Counter
import re
import os


class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.id_to_subword = {}
        self.merges = []
        self.unk_token = "<unk>"

    def normalize(self, corpus):
        corpus = corpus.lower()
        corpus = re.sub(r"\s+", " ", corpus)
        corpus = re.sub(r"[^\w\s]", "", corpus)
        return corpus

    def get_initial_tokens(self, corpus):
        tokens = corpus.split()
        return [" ".join(list(token)) + " </w>" for token in tokens]

    def get_pair_counts(self, tokenized):
        pairs = Counter()
        for word in tokenized:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    def merge_pair(self, tokenized_words, pair):
        merged = []
        bigram = " ".join(pair)
        replacement = "".join(pair)
        pattern = re.compile(r"(?<!\S)" + re.escape(bigram) + r"(?!\S)")
        for word in tokenized_words:
            merged.append(pattern.sub(replacement, word))
        return merged

    def train(self, corpus, verbose=True):
        corpus = self.normalize(corpus)
        tokenized = self.get_initial_tokens(corpus)

        # Initial subwords: all single chars + </w>
        initial_subwords = set()
        for word in tokenized:
            initial_subwords.update(word.split())

        # Add <unk> token
        initial_subwords.add(self.unk_token)

        if verbose:
            print(f"Initial subword count: {len(initial_subwords)}")

        # Maximum merges allowed
        max_merges = self.vocab_size - len(initial_subwords)
        if verbose:
            print(f"Max merges allowed: {max_merges}")

        merges_done = 0
        while merges_done < max_merges:
            pairs = self.get_pair_counts(tokenized)
            if not pairs:
                break

            most_common, freq = pairs.most_common(1)[0]
            tokenized = self.merge_pair(tokenized, most_common)
            self.merges.append(most_common)
            merges_done += 1

            if verbose and merges_done % 50 == 0:
                print(f"Merges done: {merges_done}, current most common: {most_common}")

        # Final vocab: ensure all initial subwords + merged subwords
        final_subwords = set()
        for word in tokenized:
            final_subwords.update(word.split())
        final_subwords.update(initial_subwords)  # keep single chars + <unk>

        self.vocab = {sw: idx for idx, sw in enumerate(sorted(final_subwords))}
        self.id_to_subword = {idx: sw for sw, idx in self.vocab.items()}

        if verbose:
            print(f"Training finished. Final vocab size: {len(self.vocab)}")

    def encode_word(self, word):
        symbols = list(word) + ["</w>"]
        i = 0
        while i < len(symbols) - 1:
            pair = (symbols[i], symbols[i + 1])
            if pair in self.merges:
                symbols[i:i + 2] = ["".join(pair)]
            else:
                i += 1

        # Map all symbols to IDs, unknowns to <unk>
        return [self.vocab.get(subword, self.vocab[self.unk_token]) for subword in symbols]

    def encode(self, text):
        text = self.normalize(text)
        tokens = text.split()
        encoded = []
        for token in tokens:
            encoded.extend(self.encode_word(token))
        return encoded

    def decode(self, encoded_tokens):
        subwords = [self.id_to_subword.get(idx, self.unk_token) for idx in encoded_tokens]
        text = "".join(sw.replace("</w>", " ") for sw in subwords)
        return text.strip()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BPE Tokenizer")
    parser.add_argument("files", nargs="+", help="List of text files to process")
    args = parser.parse_args()
    files = args.files

    corpus = ""
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            corpus += f.read() + "\n"

    tokenizer = BPETokenizer(vocab_size=10000)
    tokenizer.train(corpus)

    os.makedirs("result", exist_ok=True)
    with open("result/bpe_vocab.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer.vocab, f)
    with open("result/bpe_merges.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer.merges, f)
    print("Vocabulary Size:", len(tokenizer.vocab))

    # Test encoding and decoding
    test = "Time flies like an arrow."
    encoded = tokenizer.encode(test)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)


if __name__ == "__main__":
    main()