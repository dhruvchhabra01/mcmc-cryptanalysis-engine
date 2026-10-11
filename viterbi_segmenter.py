#!/usr/bin/env python3
"""viterbi_segmenter.py — Optimal word segmentation via Viterbi dynamic programming.

Given a string of uppercase letters with all spaces removed (as produced by a
substitution cipher decryption attempt), this module recovers the most likely
word boundaries using a pre-computed word log-frequency dictionary.

Algorithm
---------
Classic **Viterbi / shortest-path DP** on a DAG whose nodes are character
positions ``0 … N`` and whose edges ``(i, j)`` carry the log-probability of
the substring ``text[i:j]`` being a valid English word.

*  **Time complexity** : O(N · L) where *N* = len(text), *L* = max word length.
*  **Space complexity**: O(N) for the score and back-pointer arrays.

Unknown-word handling
~~~~~~~~~~~~~~~~~~~~~
Substrings absent from the dictionary are **not** silently dropped.  Instead
they receive a heavy per-character penalty so the optimiser still prefers
known words but never gets stuck::

    score(unknown, length) = UNKNOWN_BASE_PENALTY + length * UNKNOWN_PER_CHAR

This ensures single-character fallback is always available as a worst case,
while longer unknown chunks are exponentially disfavoured relative to valid
dictionary splits.

Usage
-----
::

    import pickle
    from viterbi_segmenter import viterbi_segment

    with open("word_log_freqs.pkl", "rb") as f:
        word_dict = pickle.load(f)

    plain = "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMES"
    segmented, score = viterbi_segment(plain, word_dict)
    print(segmented)
    print(score)
"""

from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Additive penalty for an out-of-vocabulary (OOV) token.
UNKNOWN_BASE_PENALTY: float = -25.0

#: Per-character penalty stacked on top of the base for OOV tokens.
#: Makes longer unknown chunks far more expensive than short ones,
#: steering the algorithm toward single-char fallback when no word fits.
UNKNOWN_PER_CHAR: float = -5.0

#: Absolute cap on the word lengths the algorithm considers.  Most
#: English words are ≤ 20 characters; capping avoids pointless inner
#: iterations on very long inputs.
MAX_WORD_LENGTH_CAP: int = 28


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _effective_max_word_length(word_dict: Dict[str, float]) -> int:
    """Return the maximum key length in *word_dict*, capped for safety.

    Parameters
    ----------
    word_dict : dict[str, float]
        Word → log-frequency mapping (keys are lowercase).

    Returns
    -------
    int
        ``min(max(len(w) for w in word_dict), MAX_WORD_LENGTH_CAP)``.
    """
    if not word_dict:
        return 1
    longest = max(len(w) for w in word_dict)
    return min(longest, MAX_WORD_LENGTH_CAP)


def _oov_score(length: int) -> float:
    """Return the penalty score for an out-of-vocabulary token of *length*.

    Parameters
    ----------
    length : int
        Number of characters in the unknown chunk (≥ 1).

    Returns
    -------
    float
        A strongly negative score that grows worse with *length*.
    """
    return UNKNOWN_BASE_PENALTY + length * UNKNOWN_PER_CHAR


def viterbi_segment(
    text: str,
    word_dict: Dict[str, float],
    *,
    unknown_base_penalty: float = UNKNOWN_BASE_PENALTY,
    unknown_per_char: float = UNKNOWN_PER_CHAR,
) -> Tuple[str, float]:
    """Segment a space-free uppercase string into the most probable English words.

    Implements the Viterbi dynamic-programming algorithm over a left-to-right
    DAG of candidate word spans.  The objective is to maximise the sum of
    log-probabilities (equivalently, maximise the product of probabilities)
    over all possible segmentations.

    Parameters
    ----------
    text : str
        Input string of uppercase ASCII letters with **no** spaces or
        punctuation.  The function lowercases internally; passing lowercase
        or mixed-case text is also fine.
    word_dict : dict[str, float]
        Mapping of lowercase English words to their **log-frequency**
        (as produced by :func:`build_stats.compute_word_log_freqs`).
    unknown_base_penalty : float, optional
        Fixed additive penalty for each OOV token (default -25.0).
    unknown_per_char : float, optional
        Additional penalty per character for OOV tokens (default -5.0).

    Returns
    -------
    tuple[str, float]
        ``(segmented_text, total_score)`` where *segmented_text* is the
        lowercase, space-separated best segmentation and *total_score* is
        the sum of log-probabilities along the optimal path.

    Raises
    ------
    ValueError
        If *text* is empty.

    Examples
    --------
    >>> d = {"it": -4.0, "was": -5.0, "the": -3.0, "best": -7.0}
    >>> seg, sc = viterbi_segment("ITWASTHEBEST", d)
    >>> seg
    'it was the best'

    Complexity
    ----------
    *  **Time** : O(N · L)  with N = len(text), L = max word length.
    *  **Space**: O(N)       for the DP and back-pointer arrays.
    """
    if not text:
        raise ValueError("Input text must be non-empty.")

    lower: str = text.lower()
    n: int = len(lower)

    # Pre-compute the effective max word length once.
    max_len: int = _effective_max_word_length(word_dict)

    # DP arrays ----------------------------------------------------------
    # best[i] = best (highest) total log-prob for segmenting lower[:i].
    # back[i] = length of the last word in that optimal segmentation.
    neg_inf: float = -math.inf
    best: List[float] = [neg_inf] * (n + 1)
    back: List[int] = [0] * (n + 1)
    best[0] = 0.0  # empty prefix has probability 1 → log-prob 0

    # Forward pass -------------------------------------------------------
    for i in range(1, n + 1):
        # Try every candidate word ending at position i.
        # Word length l ranges from 1 to min(max_len, i).
        limit: int = min(max_len, i)

        for l in range(1, limit + 1):
            j: int = i - l                  # start index of candidate word
            candidate: str = lower[j:i]
            score: float = word_dict.get(candidate, neg_inf)

            if score == neg_inf:
                # OOV: apply heavy penalty, but only if l == 1 so that
                # multi-char unknown runs are built from single-char
                # fallback pieces (this is cheaper than one big chunk).
                if l == 1:
                    score = unknown_base_penalty + unknown_per_char
                else:
                    # Also allow multi-char OOV so the algorithm can still
                    # greedily grab a longer unknown chunk if it ever beats
                    # the single-char cascade (unlikely but safe).
                    score = unknown_base_penalty + l * unknown_per_char

            total: float = best[j] + score
            if total > best[i]:
                best[i] = total
                back[i] = l

    # Back-trace ---------------------------------------------------------
    words: List[str] = []
    idx: int = n
    while idx > 0:
        word_len: int = back[idx]
        words.append(lower[idx - word_len : idx])
        idx -= word_len

    words.reverse()
    segmented: str = " ".join(words)
    total_score: float = best[n]

    return segmented, total_score


# ---------------------------------------------------------------------------
# Convenience: command-line quick test
# ---------------------------------------------------------------------------


def main() -> None:
    """Quick smoke test — segment a hard-coded example or argv[1]."""
    pkl_path = Path(__file__).resolve().parent / "word_log_freqs.pkl"
    if not pkl_path.exists():
        print(
            f"ERROR: {pkl_path} not found.  Run build_stats.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(pkl_path, "rb") as fh:
        word_dict: Dict[str, float] = pickle.load(fh)

    test_inputs: List[str] = [
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMES",
        "TOBEORNOTTOBETHATISTHEQUESTION",
        "ALICEWASBEGINNINGTOGETVERYTIREDOFSITTING",
        "ELEMENTARYDEARMYWATSONSAIDHOLMES",
        "CALLMEISHMAEL",
    ]

    if len(sys.argv) > 1:
        test_inputs = [sys.argv[1].upper()]

    max_label = max(len(t) for t in test_inputs)
    for ciphertext in test_inputs:
        segmented, score = viterbi_segment(ciphertext, word_dict)
        print(f"  Input : {ciphertext}")
        print(f"  Output: {segmented}")
        print(f"  Score : {score:+.4f}")
        print()


if __name__ == "__main__":
    main()
