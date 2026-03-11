# AI Tokenizer

This repository contains a collection of custom implementation of tokenizers for text processing.

## Features

- **Word Tokenizer** – splits text into individual words.
- **BPE Tokenizer** – implements Byte-Pair Encoding (BPE) subword tokenization.
- Supports **mono-CPU** and **multi-CPU parallel** implementations.

## Multi-CPU BPE Notes

The multi-CPU (parallel) approach attempts to distribute bigram counting and merging across multiple CPU cores.  
However, for BPE training:

- Most operations are highly sequential because each merge changes the subwords for the next iteration.
- CPU parallelization adds overhead for splitting, merging, and collecting results between processes.

In practice, this makes the parallel version slower than the single-threaded version for small to medium corpora.

## Installation

No additional packages are required. Clone the repository:

```bash
git clone https://github.com/coderanger428/ai-tokenizer.git
cd ai-tokenizer
```

## Usage

### Run BPE Tokenizer (mono-CPU)

```bash
python bpe_tokenizer.py <corpus file(s)>
```

### Run BPE Tokenizer (multi-CPU parallel)

```bash
python bpe_parallel.py <corpus file(s)>
```

### Run Simple Word Tokenizer

```bash
python word_tokenizer.py <corpus file(s)>
```

Output files (vocab, merges) are stored in the `output/` directory by default.

## Notes

- Increasing vocab size may be necessary to avoid subword splitting issues (e.g., `like` being broken into individual letters).
- Single-character tokens are always included in the vocab to prevent any character from being lost.
- Parallelization is limited by the sequential nature of BPE merges, so multi-CPU may not improve runtime significantly for moderate corpora.
