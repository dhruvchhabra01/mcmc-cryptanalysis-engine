#!/usr/bin/env python3
"""build_stats.py — Corpus-derived English language statistics for MCMC cryptanalysis.

Downloads several public-domain books from Project Gutenberg, strips the
boilerplate headers/footers, and computes two statistical artefacts:

1.  **Bigram log-probability matrix** (26×26, NumPy float64)
    Entry ``M[i, j]`` = log P(letter *j* follows letter *i*), estimated via
    Laplace-smoothed counts from the corpus.

2.  **Word log-frequency dictionary** (top 10 000 words → float)
    Maps each word to ``log(count / total_words)`` so that word-level scoring
    reduces to simple addition.

Both artefacts are serialised with :mod:`pickle` and written next to this
script as ``bigram_log_probs.pkl`` and ``word_log_freqs.pkl``.

Usage
-----
::

    $ python build_stats.py          # downloads, computes, saves
    $ python build_stats.py --help   # prints this docstring

Requirements
------------
* ``requests``
* ``numpy``
"""

from __future__ import annotations

import logging
import math
import pickle
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Project Gutenberg mirrors — plain-text URLs for several classic English books.
# Using multiple books gives a richer bigram / word-frequency estimate.
GUTENBERG_URLS: Dict[str, str] = {
    "Alice's Adventures in Wonderland": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "Pride and Prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "A Tale of Two Cities": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "The Adventures of Sherlock Holmes": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "Moby Dick": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "Great Expectations": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
}

ALPHABET: str = string.ascii_lowercase          # 'abcdefghijklmnopqrstuvwxyz'
NUM_LETTERS: int = len(ALPHABET)                 # 26
CHAR_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(ALPHABET)}

TOP_K_WORDS: int = 10_000

OUTPUT_DIR: Path = Path(__file__).resolve().parent
BIGRAM_PKL: Path = OUTPUT_DIR / "bigram_log_probs.pkl"
WORD_FREQ_PKL: Path = OUTPUT_DIR / "word_log_freqs.pkl"

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_text(url: str, title: str, *, timeout: int = 30) -> str:
    """Download a plain-text file from *url* and return its content.

    Parameters
    ----------
    url : str
        Direct URL to a UTF-8 plain-text resource.
    title : str
        Human-readable title used only for logging.
    timeout : int, optional
        HTTP request timeout in seconds (default 30).

    Returns
    -------
    str
        The decoded text body.

    Raises
    ------
    requests.HTTPError
        If the server responds with a non-2xx status code.
    """
    LOG.info("Downloading '%s' from %s …", title, url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Remove Project Gutenberg header and footer from *raw* text.

    The function looks for the conventional ``*** START OF …`` and
    ``*** END OF …`` markers.  If they are absent the full text is
    returned unchanged (with a warning).

    Parameters
    ----------
    raw : str
        Full text including Gutenberg boilerplate.

    Returns
    -------
    str
        The literary content between the markers.
    """
    start_markers = [
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
        r"\*\*\* START OF THE PROJECT GUTENBERG",
    ]
    end_markers = [
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
        r"\*\*\* END OF THE PROJECT GUTENBERG",
    ]

    start_idx = 0
    for pat in start_markers:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            # Advance past the marker line.
            start_idx = raw.index("\n", m.end()) + 1
            break
    else:
        LOG.warning("Could not find Gutenberg START marker — using full text.")

    end_idx = len(raw)
    for pat in end_markers:
        m = re.search(pat, raw[start_idx:], re.IGNORECASE)
        if m:
            end_idx = start_idx + m.start()
            break
    else:
        LOG.warning("Could not find Gutenberg END marker — using text to EOF.")

    return raw[start_idx:end_idx]


def fetch_corpus(urls: Dict[str, str] | None = None) -> str:
    """Download and concatenate multiple Gutenberg texts into one corpus.

    Parameters
    ----------
    urls : dict[str, str] | None
        Mapping of ``{title: url}``.  Defaults to :data:`GUTENBERG_URLS`.

    Returns
    -------
    str
        Cleaned, concatenated corpus text.
    """
    if urls is None:
        urls = GUTENBERG_URLS
    parts: List[str] = []
    for title, url in urls.items():
        try:
            raw = download_text(url, title)
            clean = strip_gutenberg_boilerplate(raw)
            parts.append(clean)
            LOG.info("  → %s: %d characters", title, len(clean))
        except requests.RequestException as exc:
            LOG.error("Failed to download '%s': %s", title, exc)
    if not parts:
        raise RuntimeError("Could not download any corpus texts.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Bigram statistics
# ---------------------------------------------------------------------------

_ALPHA_RE = re.compile(r"[a-z]")


def compute_bigram_log_probs(
    text: str,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Build a 26×26 Laplace-smoothed log-probability bigram matrix.

    For every ordered pair of letters ``(a, b)`` the matrix entry
    ``M[i(a), i(b)]`` equals:

    .. math::

        \\log \\frac{C(a, b) + \\alpha}{\\sum_{b'} C(a, b') + 26\\,\\alpha}

    where *C(a, b)* is the raw count of the bigram *ab* in *text* and
    *α* is the Laplace smoothing parameter.

    Parameters
    ----------
    text : str
        Input text (case-insensitive; non-alphabetic characters are
        treated as word boundaries and do **not** produce bigrams).
    alpha : float, optional
        Laplace (additive) smoothing constant.  Default ``1.0``.

    Returns
    -------
    numpy.ndarray
        Shape ``(26, 26)`` array of ``float64`` log-probabilities.
    """
    counts = np.zeros((NUM_LETTERS, NUM_LETTERS), dtype=np.float64)

    lower = text.lower()
    prev_idx: int | None = None

    for ch in lower:
        if ch in CHAR_TO_IDX:
            cur_idx = CHAR_TO_IDX[ch]
            if prev_idx is not None:
                counts[prev_idx, cur_idx] += 1
            prev_idx = cur_idx
        else:
            # Non-alpha character breaks the chain.
            prev_idx = None

    # Laplace smoothing: add α to every cell.
    smoothed = counts + alpha

    # Normalise each row to a probability distribution, then take log.
    row_sums = smoothed.sum(axis=1, keepdims=True)  # (26, 1)
    log_probs: np.ndarray = np.log(smoothed / row_sums)

    LOG.info(
        "Bigram matrix built — total raw bigram tokens: %s",
        f"{int(counts.sum()):,}",
    )
    return log_probs


# ---------------------------------------------------------------------------
# Word frequency statistics
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def compute_word_log_freqs(
    text: str,
    *,
    top_k: int = TOP_K_WORDS,
) -> Dict[str, float]:
    """Return the *top_k* most frequent words mapped to log-frequencies.

    A "word" is defined as a maximal run of lowercase ASCII letters,
    optionally containing a single internal apostrophe (e.g. ``don't``).

    The log-frequency of word *w* is:

    .. math::

        \\log \\frac{\\text{count}(w)}{N}

    where *N* is the total token count across the **entire** vocabulary
    (not just the top-*k* slice).

    Parameters
    ----------
    text : str
        Input text (case-insensitive).
    top_k : int, optional
        Number of most-common words to retain.  Default ``10_000``.

    Returns
    -------
    dict[str, float]
        ``{word: log_frequency}`` for the *top_k* words.
    """
    lower = text.lower()
    tokens: List[str] = _WORD_RE.findall(lower)
    total: int = len(tokens)

    if total == 0:
        raise ValueError("No words found in the supplied text.")

    counter = Counter(tokens)
    most_common: List[Tuple[str, int]] = counter.most_common(top_k)

    log_total = math.log(total)
    freqs: Dict[str, float] = {
        word: math.log(count) - log_total for word, count in most_common
    }

    LOG.info(
        "Word frequencies built — %s total tokens, %s unique, keeping top %s",
        f"{total:,}",
        f"{len(counter):,}",
        f"{len(freqs):,}",
    )
    return freqs


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_pickle(obj: object, path: Path) -> None:
    """Serialise *obj* to *path* using :mod:`pickle` protocol 5.

    Parameters
    ----------
    obj : object
        Any picklable Python object.
    path : pathlib.Path
        Destination file path.
    """
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=5)
    LOG.info("Saved %s (%s bytes)", path.name, f"{path.stat().st_size:,}")


# ---------------------------------------------------------------------------
# Inspection / summary helpers
# ---------------------------------------------------------------------------


def summarise_bigram_matrix(M: np.ndarray) -> str:
    """Return a human-readable summary of the bigram matrix *M*.

    Parameters
    ----------
    M : numpy.ndarray
        26×26 log-probability matrix.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = [
        f"Bigram log-probability matrix  shape={M.shape}  dtype={M.dtype}",
        f"  min={M.min():.4f}   max={M.max():.4f}   mean={M.mean():.4f}",
    ]
    # Show the top-5 most probable bigrams.
    flat = M.flatten()
    top5_idx = np.argsort(flat)[-5:][::-1]
    lines.append("  Top-5 bigrams:")
    for idx in top5_idx:
        r, c = divmod(idx, NUM_LETTERS)
        lines.append(
            f"    {ALPHABET[r]}{ALPHABET[c]}  log-prob={flat[idx]:+.4f}"
        )
    return "\n".join(lines)


def summarise_word_freqs(freqs: Dict[str, float], n: int = 20) -> str:
    """Return a summary showing the *n* highest-frequency words.

    Parameters
    ----------
    freqs : dict[str, float]
        Word → log-frequency mapping.
    n : int, optional
        Number of words to display.

    Returns
    -------
    str
        Multi-line summary string.
    """
    sorted_words = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)
    lines = [f"Word log-frequencies — {len(freqs):,} entries, top {n}:"]
    for word, lf in sorted_words[:n]:
        lines.append(f"  {word:<15s} {lf:+.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Download corpus, compute statistics, and persist to disk."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    LOG.info("Fetching corpus...")
    corpus = fetch_corpus()

    LOG.info("Computing statistics...")
    bigram_matrix = compute_bigram_log_probs(corpus)
    word_freqs = compute_word_log_freqs(corpus, top_k=TOP_K_WORDS)

    LOG.info("Serializing artifacts to disk...")
    save_pickle(bigram_matrix, BIGRAM_PKL)
    save_pickle(word_freqs, WORD_FREQ_PKL)

    print()
    print(summarise_bigram_matrix(bigram_matrix))
    print()
    print(summarise_word_freqs(word_freqs))
    print()
    LOG.info("Statistics generated successfully: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
