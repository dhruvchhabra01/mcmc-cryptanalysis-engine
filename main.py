#!/usr/bin/env python3
"""main.py — Master MCMC cryptanalysis engine for space-stripped substitution ciphers.

Fuses the bigram-based MCMC sampler (:mod:`cipher_core`) with the Viterbi
word segmenter (:mod:`viterbi_segmenter`) into a two-stage scoring pipeline
that can decrypt ciphertext *even when all spaces have been removed*.

Scoring pipeline (short-circuit)
--------------------------------
1.  **Bigram gate** — Score the candidate plaintext using the fast vectorised
    26×26 bigram log-probability matrix.  A rolling window of the top 100
    bigram scores is maintained.  If the current score falls more than 1σ
    below that window's mean, the candidate is **rejected immediately**
    without running the expensive Viterbi pass.

2.  **Viterbi fusion** — If the bigram score clears the gate, the plaintext
    is fed through :func:`viterbi_segmenter.viterbi_segment` to obtain a
    word-level log-probability.

3.  **Weighted combination** — The final energy used in the Metropolis
    acceptance step is::

        E = α · bigram_score  +  β · viterbi_score

    where *α* and *β* are tuneable weights (defaults: α = 1.0, β = 0.5).

Usage
-----
::

    $ python main.py                          # built-in 50-word demo
    $ python main.py "CIPHERTEXT"             # decrypt user input
    $ python main.py --help

Requirements
------------
* ``numpy``
* ``bigram_log_probs.pkl`` and ``word_log_freqs.pkl`` (from :mod:`build_stats`)
"""

from __future__ import annotations

import math
import pickle
import random
import string
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Sibling module imports
# ---------------------------------------------------------------------------
from cipher_core import (
    ALPHABET,
    CHAR_TO_IDX,
    NUM_LETTERS,
    _key_to_map,
    _precompute_indices,
    decrypt,
    encrypt,
    random_key,
    score_fast,
    swap_two,
)
from viterbi_segmenter import viterbi_segment

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_ITERATIONS: int = 50_000
DEFAULT_T0: float = 2.0
DEFAULT_COOLING: float = 0.9998
DEFAULT_PRINT_EVERY: int = 5_000

#: Weighted-combination coefficients: E = α·bigram + β·viterbi
DEFAULT_ALPHA: float = 1.0      # bigram weight
DEFAULT_BETA: float = 0.5       # viterbi / word-frequency weight

#: Rolling window size for the bigram short-circuit gate.
ROLLING_WINDOW: int = 100

#: How many σ below the rolling mean triggers early rejection.
GATE_SIGMA: float = 1.0

#: Asset paths (next to this script).
_PKL_DIR: Path = Path(__file__).resolve().parent
BIGRAM_PKL: Path = _PKL_DIR / "bigram_log_probs.pkl"
WORD_FREQ_PKL: Path = _PKL_DIR / "word_log_freqs.pkl"


# ---------------------------------------------------------------------------
# Rolling statistics tracker
# ---------------------------------------------------------------------------


class RollingGate:
    """Maintains a fixed-size window of recent scores for short-circuit gating.

    The gate passes a candidate score *s* iff::

        s  ≥  mean(window) − sigma_threshold × std(window)

    Until the window is full the gate always passes (warm-up phase).

    Parameters
    ----------
    window_size : int
        Number of scores to keep in the rolling window.
    sigma_threshold : float
        Number of standard deviations below the mean to set the cutoff.
    """

    def __init__(
        self,
        window_size: int = ROLLING_WINDOW,
        sigma_threshold: float = GATE_SIGMA,
    ) -> None:
        self._window: Deque[float] = deque(maxlen=window_size)
        self._sigma: float = sigma_threshold

    # -- public API -------------------------------------------------------

    def update(self, score: float) -> None:
        """Push *score* into the rolling window."""
        self._window.append(score)

    def passes(self, score: float) -> bool:
        """Return ``True`` if *score* clears the gate.

        During the warm-up phase (window not yet full) the gate always
        returns ``True`` to avoid starving the Viterbi scorer of samples.

        Parameters
        ----------
        score : float
            Candidate bigram log-likelihood.

        Returns
        -------
        bool
        """
        if len(self._window) < self._window.maxlen:
            return True                                     # warm-up
        arr = np.asarray(self._window)
        mean = arr.mean()
        std = arr.std()
        threshold = mean - self._sigma * std
        return bool(score >= threshold)

    @property
    def stats(self) -> Tuple[float, float]:
        """Return ``(mean, std)`` of the current window (or (0, 0) if empty)."""
        if not self._window:
            return 0.0, 0.0
        arr = np.asarray(self._window)
        return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Master MCMC engine
# ---------------------------------------------------------------------------


def mcmc_decrypt_fused(
    ciphertext: str,
    bigram_log_probs: np.ndarray,
    word_dict: Dict[str, float],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    t0: float = DEFAULT_T0,
    cooling: float = DEFAULT_COOLING,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    print_every: int = DEFAULT_PRINT_EVERY,
    seed: int | None = None,
    initial_key: List[str] | None = None,
    verbose: bool = True,
) -> Tuple[str, str, List[str], float]:
    """Decrypt a space-stripped substitution cipher using bigram + Viterbi fusion.

    This is the master engine that combines:

    *  Fast vectorised bigram scoring with a rolling short-circuit gate.
    *  Viterbi optimal word segmentation for candidates that pass the gate.
    *  Simulated-annealing Metropolis–Hastings acceptance.

    Parameters
    ----------
    ciphertext : str
        Encrypted text with **all spaces removed** (uppercase letters only).
    bigram_log_probs : numpy.ndarray
        26×26 Laplace-smoothed bigram log-probability matrix.
    word_dict : dict[str, float]
        Top-10 000 English words → log-frequency (from :mod:`build_stats`).
    iterations : int
        Number of MCMC steps (default 20 000).
    t0 : float
        Initial annealing temperature (default 1.0).
    cooling : float
        Multiplicative cooling rate (default 0.9995).
    alpha : float
        Weight for the bigram score component (default 1.0).
    beta : float
        Weight for the Viterbi word-frequency score component (default 0.5).
    print_every : int
        Progress snapshot interval (default 1 000). 0 to suppress.
    seed : int | None
        Random seed for reproducibility.
    initial_key : list[str] | None
        Optional starting key (warm start).  If ``None``, a random key is used.
    verbose : bool
        If ``True``, print convergence trace to stdout.

    Returns
    -------
    tuple[str, str, list[str], float]
        ``(best_segmented, best_raw_plain, best_key, best_combined_score)``

        *  ``best_segmented`` — space-separated plaintext from Viterbi.
        *  ``best_raw_plain`` — raw lowercase decryption (no spaces).
        *  ``best_key`` — the 26-letter substitution key.
        *  ``best_combined_score`` — α·bigram + β·viterbi.
    """
    rng = random.Random(seed)
    t_start = time.perf_counter()

    # -- Pre-compute cipher bigram index pairs (done once) ----------------
    c_first, c_second = _precompute_indices(ciphertext)

    # -- Initialise key ---------------------------------------------------
    current_key: List[str] = list(initial_key) if initial_key else random_key(rng)
    current_map: np.ndarray = _key_to_map(current_key)

    current_bigram: float = score_fast(
        c_first, c_second, current_map, bigram_log_probs,
    )
    current_plain: str = decrypt(ciphertext, current_key)
    _, current_viterbi = viterbi_segment(current_plain, word_dict)
    current_combined: float = alpha * current_bigram + beta * current_viterbi

    # -- Best tracking ----------------------------------------------------
    best_key: List[str] = current_key.copy()
    best_combined: float = current_combined
    best_bigram: float = current_bigram
    best_viterbi: float = current_viterbi
    best_plain: str = current_plain

    # -- Rolling gate for short-circuit evaluation ------------------------
    gate = RollingGate(window_size=ROLLING_WINDOW, sigma_threshold=GATE_SIGMA)
    gate.update(current_bigram)

    T: float = t0
    accepted: int = 0
    short_circuited: int = 0

    if verbose and print_every > 0:
        _print_header()

    # =====================================================================
    # Main MCMC loop
    # =====================================================================
    for step in range(1, iterations + 1):

        # 1. Propose neighbour key ----------------------------------------
        new_key: List[str] = swap_two(current_key, rng)
        new_map: np.ndarray = _key_to_map(new_key)

        # 2. Bigram score (cheap) -----------------------------------------
        new_bigram: float = score_fast(
            c_first, c_second, new_map, bigram_log_probs,
        )

        # 3. Short-circuit gate -------------------------------------------
        if not gate.passes(new_bigram):
            short_circuited += 1
            T *= cooling
            # Print progress even on short-circuit steps.
            if verbose and print_every > 0 and step % print_every == 0:
                best_seg, _ = viterbi_segment(best_plain, word_dict)
                _print_progress(
                    step=step,
                    temperature=T,
                    bigram_score=best_bigram,
                    viterbi_score=best_viterbi,
                    combined_score=best_combined,
                    acceptance_rate=accepted / step,
                    short_circuit_rate=short_circuited / step,
                    preview=best_seg,
                )
            continue

        # 4. Viterbi score (expensive — only if gate passed) --------------
        new_plain: str = decrypt(ciphertext, new_key)
        _, new_viterbi = viterbi_segment(new_plain, word_dict)
        new_combined: float = alpha * new_bigram + beta * new_viterbi

        # 5. Metropolis acceptance ----------------------------------------
        delta: float = new_combined - current_combined
        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-30)):
            current_key = new_key
            current_map = new_map
            current_bigram = new_bigram
            current_viterbi = new_viterbi
            current_combined = new_combined
            current_plain = new_plain
            accepted += 1

            # Update rolling gate with accepted bigram scores.
            gate.update(current_bigram)

            # Track global best.
            if current_combined > best_combined:
                best_combined = current_combined
                best_bigram = current_bigram
                best_viterbi = current_viterbi
                best_key = current_key.copy()
                best_plain = current_plain

        # 6. Cool ---------------------------------------------------------
        T *= cooling

        # 7. Progress trace -----------------------------------------------
        if verbose and print_every > 0 and step % print_every == 0:
            best_seg, _ = viterbi_segment(best_plain, word_dict)
            _print_progress(
                step=step,
                temperature=T,
                bigram_score=best_bigram,
                viterbi_score=best_viterbi,
                combined_score=best_combined,
                acceptance_rate=accepted / step,
                short_circuit_rate=short_circuited / step,
                preview=best_seg,
            )

    # -- Final results ----------------------------------------------------
    elapsed = time.perf_counter() - t_start
    best_seg, best_seg_score = viterbi_segment(best_plain, word_dict)

    if verbose:
        gate_mean, gate_std = gate.stats
        print(f"\n{'=' * 80}")
        print(f"  Finished {iterations:,} iterations in {elapsed:.2f}s")
        print(f"  Final temperature   : {T:.6e}")
        print(f"  Acceptance rate     : {accepted / iterations:.2%}")
        print(f"  Short-circuit rate  : {short_circuited / iterations:.2%}")
        print(f"  Gate threshold      : {gate_mean - GATE_SIGMA * gate_std:+.2f} "
              f"(mean={gate_mean:+.2f}, std={gate_std:.2f})")
        print(f"  Best bigram score   : {best_bigram:+.4f}")
        print(f"  Best viterbi score  : {best_viterbi:+.4f}")
        print(f"  Best combined score : {best_combined:+.4f}  "
              f"(alpha={alpha}, beta={beta})")
        print(f"  Best segmentation   : {best_seg[:120]}")
        print(f"{'=' * 80}\n")

    return best_seg, best_plain, best_key, best_combined


def _print_header() -> None:
    """Print the column header for the convergence trace."""
    print(f"\n{'-' * 80}")
    print(
        f"  {'Step':>6s}  {'Temp':>9s}  {'Bigram':>9s}  {'Viterbi':>9s}  "
        f"{'Combined':>9s}  {'Acc%':>5s}  {'SC%':>5s}  Segmentation"
    )
    print(f"{'-' * 80}")


def _print_progress(
    *,
    step: int,
    temperature: float,
    bigram_score: float,
    viterbi_score: float,
    combined_score: float,
    acceptance_rate: float,
    short_circuit_rate: float,
    preview: str,
) -> None:
    """Print one row of the convergence trace."""
    max_preview = 42
    short = preview[:max_preview] + ("..." if len(preview) > max_preview else "")
    print(
        f"  {step:6d}  {temperature:9.5f}  {bigram_score:+9.1f}  "
        f"{viterbi_score:+9.1f}  {combined_score:+9.1f}  "
        f"{acceptance_rate:4.1%}  {short_circuit_rate:4.1%}  {short}"
    )


# ---------------------------------------------------------------------------
# Test block — 50-word encrypt → strip spaces → crack
# ---------------------------------------------------------------------------


def _load_assets() -> Tuple[np.ndarray, Dict[str, float]]:
    """Load the pickled bigram matrix and word-frequency dictionary.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, float]]

    Raises
    ------
    SystemExit
        If either ``.pkl`` file is missing.
    """
    for p in (BIGRAM_PKL, WORD_FREQ_PKL):
        if not p.exists():
            print(
                f"ERROR: {p} not found.  Run build_stats.py first.",
                file=sys.stderr,
            )
            sys.exit(1)

    with open(BIGRAM_PKL, "rb") as f:
        bigram_log_probs: np.ndarray = pickle.load(f)
    with open(WORD_FREQ_PKL, "rb") as f:
        word_dict: Dict[str, float] = pickle.load(f)

    return bigram_log_probs, word_dict


def run_demo() -> None:
    """Encrypt a 50-word paragraph, strip spaces, and decrypt with the master engine.

    Uses a two-phase strategy:
    *  **Phase 1** — Fast bigram-only MCMC (via :mod:`cipher_core`) to narrow
       down the key.  This is fast and reliable even without spaces.
    *  **Phase 2** — Refinement using the fused bigram + Viterbi scorer,
       starting from Phase 1's best key instead of random.
    """
    from cipher_core import mcmc_decrypt

    bigram_log_probs, word_dict = _load_assets()

    # ── Test paragraph — ~500+ chars gives bigrams enough signal ──
    # (Space-stripped MCMC needs more text than the spaces-intact version)
    plaintext = (
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
    word_count = len(plaintext.split())

    # Generate a random ground-truth key.
    true_key: List[str] = random_key(random.Random(7))

    # Encrypt and strip all spaces.
    ciphertext_with_spaces = encrypt(plaintext, true_key)
    ciphertext = ciphertext_with_spaces.replace(" ", "")

    print("=" * 80)
    print("  MCMC Cryptanalysis Engine — Space-Stripped Cipher Demo")
    print("=" * 80)
    print(f"\n  Original ({word_count} words):")
    print(f"    {plaintext[:100]}...")
    print(f"\n  True key      : {''.join(true_key)}")
    print(f"  Ciphertext    : {ciphertext[:80]}...")
    print(f"  Cipher length : {len(ciphertext)} characters (no spaces)")

    # Phase 1: Fast bigram-only MCMC
    NUM_P1_RESTARTS = 10
    print(f"\n{'-' * 80}")
    print(f"  Phase 1: Bigram-only MCMC ({NUM_P1_RESTARTS} restarts x 20k iterations)")
    print(f"{'-' * 80}")

    p1_best_key: List[str] = []
    p1_best_score: float = -math.inf

    for r in range(1, NUM_P1_RESTARTS + 1):
        _, key, score = mcmc_decrypt(
            ciphertext,
            bigram_log_probs,
            iterations=20_000,
            print_every=0,
            verbose=False,
        )
        marker = " [new best]" if score > p1_best_score else ""
        print(f"    Restart {r:2d}/{NUM_P1_RESTARTS}  "
              f"score={score:+.2f}{marker}")
        if score > p1_best_score:
            p1_best_score = score
            p1_best_key = key

    p1_plain = decrypt(ciphertext, p1_best_key)
    p1_seg, _ = viterbi_segment(p1_plain, word_dict)
    p1_correct = sum(a == b for a, b in zip(true_key, p1_best_key))
    print(f"\n  Phase 1 result: {p1_correct}/26 keys, score={p1_best_score:+.2f}")
    print(f"  Segmented     : {p1_seg[:100]}...")

    # Phase 2: Fused refinement
    print(f"\n{'-' * 80}")
    print(f"  Phase 2: Fused bigram + Viterbi refinement (warm start)")
    print(f"{'-' * 80}")

    best_seg, best_raw, best_key, best_score = mcmc_decrypt_fused(
        ciphertext,
        bigram_log_probs,
        word_dict,
        iterations=30_000,
        t0=0.5,
        cooling=0.9998,
        initial_key=p1_best_key,
        verbose=True,
    )

    correct = sum(a == b for a, b in zip(true_key, best_key))
    print("=" * 80)
    print("  FINAL RESULT")
    print("=" * 80)
    print(f"  Combined score : {best_score:+.4f}")
    print(f"  Key accuracy   : {correct}/26 letters correct")
    print(f"  Key found      : {''.join(best_key)}")
    print(f"  Key actual     : {''.join(true_key)}")
    print(f"\n  Final segmentation:")
    print(f"    {best_seg}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse argv and run the appropriate mode."""
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    if len(sys.argv) > 1:
        # Decrypt user-supplied space-stripped ciphertext.
        ciphertext = sys.argv[1].upper().replace(" ", "")
        bigram_log_probs, word_dict = _load_assets()

        print(f"\n  Ciphertext ({len(ciphertext)} chars): {ciphertext[:80]}…")
        best_seg, best_raw, best_key, best_score = mcmc_decrypt_fused(
            ciphertext,
            bigram_log_probs,
            word_dict,
            verbose=True,
        )
        print(f"  Decrypted: {best_seg}")
    else:
        run_demo()


if __name__ == "__main__":
    main()
