#!/usr/bin/env python3
"""cipher_core.py — MCMC substitution-cipher cracker with simulated annealing.

Implements a Metropolis–Hastings sampler over the space of 26! monoalphabetic
substitution keys.  The energy function is the **bigram log-likelihood** of
the candidate plaintext under a pre-computed English language model.  A
simulated-annealing cooling schedule drives the chain from exploration toward
exploitation.

Algorithm sketch
----------------
1.  Start with a random permutation key and temperature *T* = 1.0.
2.  At each of 20 000 iterations:
    a.  Propose a neighbour by swapping two random letters in the key.
    b.  Decrypt the ciphertext with the proposed key.
    c.  Score the plaintext using the 26×26 bigram log-probability matrix.
    d.  Accept the proposal if ΔE > 0 (better), otherwise accept with
        probability exp(ΔE / T).
    e.  Cool: T ← T × 0.9995.
3.  Track and return the best key seen across the entire run.

Spaces (and any non-alphabetic characters) in the ciphertext are preserved
as-is — only letters are substituted.

Usage
-----
::

    $ python cipher_core.py                  # runs a built-in demo
    $ python cipher_core.py "CIPHER TEXT"    # decrypt argv[1]

Requirements
------------
* ``numpy``
* ``bigram_log_probs.pkl`` (produced by :mod:`build_stats`)
"""

from __future__ import annotations

import math
import pickle
import random
import string
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHABET: str = string.ascii_lowercase                      # 'a…z'
NUM_LETTERS: int = 26
CHAR_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR: Dict[int, str] = {i: c for i, c in enumerate(ALPHABET)}

#: Default SA hyper-parameters.
DEFAULT_ITERATIONS: int = 20_000
DEFAULT_T0: float = 1.0
DEFAULT_COOLING: float = 0.9995
DEFAULT_PRINT_EVERY: int = 1_000

#: Path to the pre-built bigram matrix.
_PKL_DIR: Path = Path(__file__).resolve().parent
BIGRAM_PKL: Path = _PKL_DIR / "bigram_log_probs.pkl"


# ---------------------------------------------------------------------------
# Key / cipher helpers
# ---------------------------------------------------------------------------


def random_key(rng: random.Random | None = None) -> List[str]:
    """Return a uniformly random permutation of the 26 lowercase letters.

    Parameters
    ----------
    rng : random.Random | None
        Optional seeded RNG for reproducibility.

    Returns
    -------
    list[str]
        A 26-element list representing the substitution key, where
        ``key[i]`` is the plaintext letter that ciphertext letter *i*
        maps to.
    """
    key = list(ALPHABET)
    (rng or random).shuffle(key)
    return key


def swap_two(key: List[str], rng: random.Random | None = None) -> List[str]:
    """Return a copy of *key* with two randomly chosen positions swapped.

    This is the minimal Metropolis proposal — it changes exactly one
    transposition in the permutation, ensuring the proposal distribution
    is symmetric.

    Parameters
    ----------
    key : list[str]
        Current 26-letter key.
    rng : random.Random | None
        Optional seeded RNG.

    Returns
    -------
    list[str]
        New key (the original is not mutated).
    """
    _rng = rng or random
    new_key = key.copy()
    i, j = _rng.sample(range(NUM_LETTERS), 2)
    new_key[i], new_key[j] = new_key[j], new_key[i]
    return new_key


def decrypt(ciphertext: str, key: List[str]) -> str:
    """Decrypt *ciphertext* using the substitution *key*.

    ``key[i]`` is the plaintext letter for ciphertext letter
    ``ALPHABET[i]``.  Non-alphabetic characters (spaces, punctuation,
    digits) are passed through unchanged.

    Parameters
    ----------
    ciphertext : str
        The encrypted message (case-insensitive; output is always lowercase).
    key : list[str]
        26-element substitution key.

    Returns
    -------
    str
        Decrypted plaintext (lowercase).
    """
    mapping: Dict[str, str] = {ALPHABET[i]: key[i] for i in range(NUM_LETTERS)}
    # Also map uppercase ciphertext letters.
    mapping.update({ALPHABET[i].upper(): key[i] for i in range(NUM_LETTERS)})
    return "".join(mapping.get(ch, ch) for ch in ciphertext)


def encrypt(plaintext: str, key: List[str]) -> str:
    """Encrypt *plaintext* using the substitution *key* (inverse mapping).

    This is the inverse of :func:`decrypt`: if ``key[i]`` maps ciphertext
    letter *i* → plaintext, then encryption maps plaintext → ciphertext via
    the reverse lookup.

    Parameters
    ----------
    plaintext : str
        Clear text to encrypt (case-insensitive; output is uppercase).
    key : list[str]
        26-element substitution key.

    Returns
    -------
    str
        Ciphertext (uppercase).
    """
    # Build inverse: plaintext_letter → ciphertext_letter
    inverse: Dict[str, str] = {key[i]: ALPHABET[i] for i in range(NUM_LETTERS)}
    lower = plaintext.lower()
    return "".join(inverse.get(ch, ch) for ch in lower).upper()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_text(
    text: str,
    bigram_log_probs: np.ndarray,
) -> float:
    """Score *text* by summing bigram log-probabilities.

    For every adjacent pair of letters (a, b) in *text* the function
    accumulates ``bigram_log_probs[idx(a), idx(b)]``.  Non-alphabetic
    characters break the bigram chain (identical to how the matrix was
    built in :func:`build_stats.compute_bigram_log_probs`).

    Parameters
    ----------
    text : str
        Candidate plaintext (case-insensitive).
    bigram_log_probs : numpy.ndarray
        Shape ``(26, 26)`` log-probability matrix.

    Returns
    -------
    float
        Total bigram log-likelihood (higher = more English-like).
    """
    total: float = 0.0
    prev_idx: int = -1                       # -1 ≡ "no previous letter"

    for ch in text:
        lower_ch = ch.lower()
        if lower_ch in CHAR_TO_IDX:
            cur_idx: int = CHAR_TO_IDX[lower_ch]
            if prev_idx >= 0:
                total += bigram_log_probs[prev_idx, cur_idx]
            prev_idx = cur_idx
        else:
            prev_idx = -1                    # non-alpha breaks chain

    return total


# ---------------------------------------------------------------------------
# Vectorised fast scorer (for hot inner loop)
# ---------------------------------------------------------------------------


def _precompute_indices(ciphertext: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert *ciphertext* into index pairs for vectorised scoring.

    Returns two 1-D int arrays of equal length:  ``(first, second)`` where
    each element pair is the index of adjacent alphabetic characters in the
    ciphertext.  Non-alpha characters act as chain breaks.

    Parameters
    ----------
    ciphertext : str
        The ciphertext (case-insensitive).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(first_indices, second_indices)`` — both ``int32``.
    """
    firsts: List[int] = []
    seconds: List[int] = []
    prev: int = -1
    for ch in ciphertext:
        lc = ch.lower()
        if lc in CHAR_TO_IDX:
            cur = CHAR_TO_IDX[lc]
            if prev >= 0:
                firsts.append(prev)
                seconds.append(cur)
            prev = cur
        else:
            prev = -1
    return np.array(firsts, dtype=np.int32), np.array(seconds, dtype=np.int32)


def score_fast(
    cipher_first: np.ndarray,
    cipher_second: np.ndarray,
    key_map: np.ndarray,
    bigram_log_probs: np.ndarray,
) -> float:
    """Vectorised bigram score using pre-computed index pairs.

    Instead of re-scanning the string on every iteration, we map the
    *ciphertext* indices through the current key permutation and perform a
    single fancy-indexed sum on the bigram matrix.

    Parameters
    ----------
    cipher_first : numpy.ndarray
        First-letter ciphertext indices (int32).
    cipher_second : numpy.ndarray
        Second-letter ciphertext indices (int32).
    key_map : numpy.ndarray
        Length-26 int32 array where ``key_map[cipher_idx] = plain_idx``.
    bigram_log_probs : numpy.ndarray
        Shape ``(26, 26)`` log-probability matrix.

    Returns
    -------
    float
        Total bigram log-likelihood.
    """
    plain_first = key_map[cipher_first]
    plain_second = key_map[cipher_second]
    return float(bigram_log_probs[plain_first, plain_second].sum())


def _key_to_map(key: List[str]) -> np.ndarray:
    """Convert a letter-list key to an int32 index-mapping array.

    Parameters
    ----------
    key : list[str]
        26-element key where ``key[i]`` is the plaintext letter for
        ciphertext letter ``ALPHABET[i]``.

    Returns
    -------
    numpy.ndarray
        Shape ``(26,)`` int32 array.
    """
    return np.array([CHAR_TO_IDX[c] for c in key], dtype=np.int32)


# ---------------------------------------------------------------------------
# MCMC + Simulated Annealing
# ---------------------------------------------------------------------------


def mcmc_decrypt(
    ciphertext: str,
    bigram_log_probs: np.ndarray,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    t0: float = DEFAULT_T0,
    cooling: float = DEFAULT_COOLING,
    print_every: int = DEFAULT_PRINT_EVERY,
    seed: int | None = None,
    verbose: bool = True,
) -> Tuple[str, List[str], float]:
    """Crack a monoalphabetic substitution cipher via MCMC + simulated annealing.

    Parameters
    ----------
    ciphertext : str
        The encrypted message (spaces and punctuation preserved).
    bigram_log_probs : numpy.ndarray
        26×26 Laplace-smoothed log-probability matrix produced by
        :func:`build_stats.compute_bigram_log_probs`.
    iterations : int, optional
        Number of MCMC steps (default 20 000).
    t0 : float, optional
        Initial annealing temperature (default 1.0).
    cooling : float, optional
        Multiplicative cooling factor applied every step (default 0.9995).
    print_every : int, optional
        Print a progress snapshot every *print_every* iterations
        (default 1 000).  Set to 0 to suppress all progress output.
    seed : int | None, optional
        Random seed for reproducibility.
    verbose : bool, optional
        If ``True`` (default), print the convergence trace.

    Returns
    -------
    tuple[str, list[str], float]
        ``(best_plaintext, best_key, best_score)`` — the highest-scoring
        decryption observed during the entire run.

    Notes
    -----
    The acceptance criterion is the standard Metropolis–Hastings rule with
    an energy function *E* = −(bigram log-likelihood) and a linearly
    cooled temperature schedule::

        ΔE  = new_score − current_score
        if ΔE > 0          → always accept
        elif random() < exp(ΔE / T) → accept (Boltzmann escape)
        else                → reject

    As *T → 0* the chain freezes onto a (near-)optimal key.
    """
    rng = random.Random(seed)
    t_start = time.perf_counter()

    # --- Pre-compute bigram index pairs for the ciphertext ---------------
    c_first, c_second = _precompute_indices(ciphertext)

    # --- Initialise key --------------------------------------------------
    current_key: List[str] = random_key(rng)
    current_map: np.ndarray = _key_to_map(current_key)
    current_score: float = score_fast(c_first, c_second, current_map,
                                      bigram_log_probs)

    best_key: List[str] = current_key.copy()
    best_score: float = current_score

    T: float = t0
    accepted: int = 0

    if verbose and print_every > 0:
        _print_header()

    # --- Main MCMC loop --------------------------------------------------
    for step in range(1, iterations + 1):
        # 1. Propose a neighbour key by swapping two letters.
        new_key: List[str] = swap_two(current_key, rng)
        new_map: np.ndarray = _key_to_map(new_key)
        new_score: float = score_fast(c_first, c_second, new_map,
                                      bigram_log_probs)

        # 2. Metropolis acceptance rule.
        delta: float = new_score - current_score
        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-30)):
            current_key = new_key
            current_map = new_map
            current_score = new_score
            accepted += 1

            # Track global best.
            if current_score > best_score:
                best_score = current_score
                best_key = current_key.copy()

        # 3. Cool the temperature.
        T *= cooling

        # 4. Progress trace.
        if verbose and print_every > 0 and step % print_every == 0:
            best_plain = decrypt(ciphertext, best_key)
            _print_progress(
                step=step,
                temperature=T,
                current_score=current_score,
                best_score=best_score,
                acceptance_rate=accepted / step,
                preview=best_plain,
            )

    # --- Final summary ---------------------------------------------------
    elapsed = time.perf_counter() - t_start
    best_plaintext: str = decrypt(ciphertext, best_key)

    if verbose:
        print(f"\n{'-' * 72}")
        print(f"  Finished {iterations:,} iterations in {elapsed:.2f}s")
        print(f"  Final temperature : {T:.6e}")
        print(f"  Acceptance rate   : {accepted / iterations:.2%}")
        print(f"  Best score        : {best_score:+.4f}")
        print(f"  Best plaintext    : {best_plaintext[:120]}")
        print(f"{'-' * 72}\n")

    return best_plaintext, best_key, best_score


def _print_header() -> None:
    """Print the column header for the convergence trace."""
    print(f"\n{'-' * 72}")
    print(
        f"  {'Step':>6s}  {'Temp':>10s}  {'CurScore':>10s}  "
        f"{'BestScore':>10s}  {'Accept%':>7s}  Preview"
    )
    print(f"{'-' * 72}")


def _print_progress(
    *,
    step: int,
    temperature: float,
    current_score: float,
    best_score: float,
    acceptance_rate: float,
    preview: str,
) -> None:
    """Print one row of the convergence trace."""
    max_preview = 50
    short = preview[:max_preview] + ("..." if len(preview) > max_preview else "")
    print(
        f"  {step:6d}  {temperature:10.6f}  {current_score:+10.2f}  "
        f"{best_score:+10.2f}  {acceptance_rate:6.1%}  {short}"
    )


def _demo_roundtrip(bigram_log_probs: np.ndarray, plaintext: str) -> None:
    """Encrypt *plaintext* with a random key, then decrypt via MCMC."""
    print("=" * 72)
    print("MCMC Substitution Cipher Cracker - Demo")
    print("=" * 72)

    true_key = random_key(random.Random(42))
    ciphertext = encrypt(plaintext, true_key)

    print(f"\nOriginal  : {plaintext[:100]}{'...' if len(plaintext)>100 else ''}")
    print(f"True key  : {''.join(true_key)}")
    print(f"Ciphertext: {ciphertext[:100]}{'...' if len(ciphertext)>100 else ''}")

    NUM_RESTARTS = 5
    print(f"\nRunning {NUM_RESTARTS} random restarts...\n")

    best_plain: str = ""
    best_key: List[str] = []
    best_score: float = -math.inf

    for restart in range(1, NUM_RESTARTS + 1):
        print(f"--- Restart {restart}/{NUM_RESTARTS} {'-' * 50}")
        plain, key, score = mcmc_decrypt(
            ciphertext,
            bigram_log_probs,
            iterations=DEFAULT_ITERATIONS,
            seed=None,
        )
        if score > best_score:
            best_plain, best_key, best_score = plain, key, score

    correct = sum(a == b for a, b in zip(true_key, best_key))
    print("=" * 72)
    print("OVERALL BEST (across all restarts)")
    print("=" * 72)
    print(f"Best score  : {best_score:+.4f}")
    print(f"Key accuracy: {correct}/26 letters correct")
    print(f"Key found   : {''.join(best_key)}")
    print(f"Key actual  : {''.join(true_key)}")
    print(f"Plaintext   : {best_plain[:120]}")
    print()


def main() -> None:
    """Entry point: load the bigram matrix and run a demo or decrypt argv[1]."""
    if not BIGRAM_PKL.exists():
        print(
            f"ERROR: {BIGRAM_PKL} not found. Run build_stats.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(BIGRAM_PKL, "rb") as fh:
        bigram_log_probs: np.ndarray = pickle.load(fh)

    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        # Decrypt user-supplied ciphertext.
        ciphertext = sys.argv[1]
        print(f"\n  Ciphertext: {ciphertext[:100]}")
        best_plain, best_key, best_score = mcmc_decrypt(
            ciphertext, bigram_log_probs,
        )
        print(f"  Best plaintext: {best_plain}")
    elif "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
    else:
        # Built-in demo with a famous passage — long enough for bigrams
        # to provide reliable signal (500+ characters recommended).
        sample = (
            "It was the best of times it was the worst of times "
            "it was the age of wisdom it was the age of foolishness "
            "it was the epoch of belief it was the epoch of incredulity "
            "it was the season of light it was the season of darkness "
            "it was the spring of hope it was the winter of despair "
            "we had everything before us we had nothing before us "
            "we were all going direct to heaven we were all going "
            "direct the other way in short the period was so far like "
            "the present period that some of its noisiest authorities "
            "insisted on its being received for good or for evil "
            "in the superlative degree of comparison only"
        )
        _demo_roundtrip(bigram_log_probs, sample)


if __name__ == "__main__":
    main()
    