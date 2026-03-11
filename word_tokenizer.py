# Load the corpus from a file
from collections import Counter
import re


def load_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        corpus = f.read()
    return corpus

# Normalize the corpus by converting to lowercase and removing punctuation
def normalize_corpus(corpus):
    import string
    corpus = corpus.lower()
    corpus = re.sub(r'\s+', ' ', corpus)  # Replace multiple whitespace with a single space
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))
    return corpus

# Word tokenization
def tokenize_corpus(corpus):
    tokens = corpus.split()
    return tokens

# Build a vocabulary from the tokens
def build_vocabulary(tokens):
    counter = Counter(tokens)

    vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common())}
    id_to_word = {idx: word for word, idx in vocab.items()}

    return vocab, id_to_word, counter

# Encode
def encode(tokens, vocab):
    return [vocab[token] for token in tokens if token in vocab]

# Decode
def decode(encoded_tokens, id_to_word):
    return [id_to_word[idx] for idx in encoded_tokens if idx in id_to_word]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Word Tokenizer')
    parser.add_argument('files', nargs='+', help='List of text files to process')
    args = parser.parse_args()
    files = args.files

    corpus = ""
    for file in files:
        corpus += load_corpus(file) + "\n"
    
    # Save merged corpus to a file for future use
    with open('corpus.txt', 'w', encoding='utf-8') as f:
        f.write(corpus)
    
    corpus = normalize_corpus(corpus)
    tokens = tokenize_corpus(corpus)
    vocab, id_to_word, counter = build_vocabulary(tokens)
    print("Vocabulary Size:", len(vocab))
    print("Most Common Words:", counter.most_common(10))

    # Save vocabulary to a JSON file
    import json
    with open('result/word_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
