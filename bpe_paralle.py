# bpe_parallel_fixed_monitor.py
import json
import re
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
import os


def normalize(corpus: str) -> str:
    corpus = corpus.lower()
    corpus = re.sub(r"\s+", " ", corpus)
    corpus = re.sub(r"[^\w\s]", "", corpus)
    return corpus


def get_initial_tokens(corpus: str):
    tokens = corpus.split()
    return [" ".join(list(token)) + " </w>" for token in tokens]


def count_bigrams(chunk):
    counter = Counter()
    for word in chunk:
        symbols = word.split()
        for i in range(len(symbols) - 1):
            counter[(symbols[i], symbols[i + 1])] += 1
    return counter


def merge_pair(chunk, pair):
    merged = []
    bigram_str = " ".join(pair)
    replacement = "".join(pair)
    pattern = re.compile(r"(?<!\S)" + re.escape(bigram_str) + r"(?!\S)")
    for word in chunk:
        merged.append(pattern.sub(replacement, word))
    return merged


def parallel_count_bigrams(tokenized, n_workers=None):
    n_workers = n_workers or cpu_count()
    size = (len(tokenized) + n_workers - 1) // n_workers
    chunks = [tokenized[i*size:(i+1)*size] for i in range(n_workers)]

    pool = Pool(n_workers)
    try:
        results = pool.map(count_bigrams, chunks)
    finally:
        pool.close()
        pool.join()

    total = Counter()
    for r in results:
        total.update(r)
    return total


def parallel_merge(tokenized, pair, n_workers=None):
    n_workers = n_workers or cpu_count()
    size = (len(tokenized) + n_workers - 1) // n_workers
    chunks = [tokenized[i*size:(i+1)*size] for i in range(n_workers)]

    with Pool(n_workers) as pool:
        merged_chunks = pool.starmap(merge_pair, [(chunk, pair) for chunk in chunks])

    merged = [w for sub in merged_chunks for w in sub]
    return merged


class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        self.id_to_subword = {}
        self.char_vocab = set()  # permanent single-char vocab

    def train(self, corpus, n_workers=None, verbose=True):
        corpus = normalize(corpus)
        tokenized = get_initial_tokens(corpus)

        # Permanent single-char vocab + </w>
        self.char_vocab = set()
        for word in tokenized:
            self.char_vocab.update(word.split())

        if verbose:
            print(f"Permanent char vocab size: {len(self.char_vocab)}")

        max_merges = self.vocab_size - len(self.char_vocab)
        merges_done = 0

        while merges_done < max_merges:
            bigram_counts = parallel_count_bigrams(tokenized, n_workers)
            if not bigram_counts:
                break

            most_common, freq = bigram_counts.most_common(1)[0]
            tokenized = parallel_merge(tokenized, most_common, n_workers)
            self.merges.append(most_common)
            merges_done += 1

            if verbose and merges_done % 50 == 0:
                print(f"[Info] Merge {merges_done}/{max_merges}: {most_common} (freq={freq})")

        # Final vocab: merged subwords + permanent chars
        merged_subwords = set()
        for word in tokenized:
            merged_subwords.update(word.split())
        final_subwords = self.char_vocab | merged_subwords

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
                symbols[i:i+2] = ["".join(pair)]
            else:
                i += 1
        return [self.vocab.get(sw, -1) for sw in symbols]

    def encode(self, text):
        text = normalize(text)
        encoded = []
        for word in text.split():
            encoded.extend(self.encode_word(word))
        return encoded

    def decode(self, ids):
        subwords = [self.id_to_subword.get(i, "<UNK>") for i in ids]
        return "".join(sw.replace("</w>", "") for sw in subwords)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel BPE Tokenizer Fixed with Reduced Monitoring")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    corpus = ""
    for f in args.files:
        with open(f, "r", encoding="utf-8") as file:
            corpus += file.read() + "\n"

    tokenizer = BPETokenizer(vocab_size=10000)  # increased vocab
    tokenizer.train(corpus, n_workers=4)

    with open("result/bpe_vocab_parallel.json", "w") as f:
        json.dump(tokenizer.vocab, f)
    with open("result/bpe_merges_parallel.json", "w") as f:
        json.dump(tokenizer.merges, f)

    test_sentence = "Time flies like an arrow."
    print("Original:", test_sentence)
    encoded = tokenizer.encode(test_sentence)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)