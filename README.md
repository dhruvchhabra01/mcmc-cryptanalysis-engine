# MCMC Cryptanalysis Engine

A statistical cryptanalysis engine in Python for breaking monoalphabetic substitution ciphers, with support for space-stripped (continuous) ciphertext using dynamic programming word segmentation.

---

## Overview

A monoalphabetic substitution cipher maps each letter of the alphabet to another under a secret bijection, spanning a key space of $26! \approx 4.03 \times 10^{26}$ possible permutations. Exhaustive search is intractable.

This project implements an optimization pipeline combining:

1. **Language Modeling**: A $26 \times 26$ Laplace-smoothed bigram log-probability transition matrix and a 10,000-word log-frequency dictionary estimated from public-domain literary corpora (Project Gutenberg).
2. **Markov Chain Monte Carlo (MCMC)**: A Metropolis-Hastings random-walk sampler with a Simulated Annealing exponential cooling schedule ($T_{k+1} = T_k \cdot \gamma$) over the symmetric group $S_{26}$.
3. **Short-Circuit Evaluation**: A statistical gating layer that maintains a rolling window of recent bigram scores ($\mu - 1\sigma$ cutoff) to immediately reject unpromising candidate keys before invoking heavier scoring routines.
4. **Viterbi Word Segmentation**: An $O(N \cdot L)$ dynamic programming segmentation algorithm that reconstructs word boundaries and scores candidate plaintexts when whitespace has been eliminated from the ciphertext.
5. **Two-Phase Annealing**: Initial broad key exploration driven by vectorized bigram log-likelihoods, followed by warm-started multi-objective refinement combining character-level bigrams and word-level frequencies:

$$E(k) = \alpha \cdot E_{\text{bigram}}(k) + \beta \cdot E_{\text{viterbi}}(k)$$

---

## Repository Structure

```
├── build_stats.py          # Corpus ingestion, bigram matrix & dictionary generator
├── cipher_core.py          # Core MCMC / Simulated Annealing substitution solver
├── viterbi_segmenter.py    # O(N*L) Viterbi dynamic programming word segmenter
├── main.py                 # Master fused engine for space-stripped ciphers
├── bigram_log_probs.pkl    # Serialized 26x26 bigram log-probability matrix
├── word_log_freqs.pkl      # Serialized top-10k word log-frequency dictionary
├── requirements.txt        # Python package dependencies
└── README.md
```

---

## Installation

Ensure Python 3.9+ is installed:

```bash
git clone <repository-url>
cd "MCMC Cryptanalysis Engine"
pip install -r requirements.txt
```

---

## Usage

### 1. Rebuild Language Model (Optional)

The repository includes pre-computed serialized models (`bigram_log_probs.pkl` and `word_log_freqs.pkl`). To rebuild them from fresh Project Gutenberg texts:

```bash
python3 build_stats.py
```

### 2. Solve Substitution Ciphers (Spaces Intact)

Run the standalone MCMC cipher cracker with random restarts:

```bash
# Run built-in benchmark
python3 cipher_core.py

# Decrypt custom ciphertext
python3 cipher_core.py "WE JXL EVT MTLE NG EWBTL..."
```

### 3. Solve Space-Stripped Ciphers

Run the master fused pipeline (two-phase MCMC + Viterbi word segmentation):

```bash
# Run built-in space-stripped benchmark
python3 main.py

# Decrypt custom continuous ciphertext
python3 main.py "DWZDVWKHEHVWRIWLPHV"
```

### 4. Standalone Viterbi Word Segmentation

Segment any spaceless uppercase text into optimal English word spans:

```bash
python3 viterbi_segmenter.py "ITWASTHEBESTOFTIMES"
```
