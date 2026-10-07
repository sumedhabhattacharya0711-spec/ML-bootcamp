"""Topic segmentation for long meetings.

A long transcript is split into parts that each fit one LLM call, cutting
where the topic changes. The topic signal is lexical cohesion, as in
TextTiling (Hearst, 1997): compare the words in a window of lines before and
after each gap; a dip in similarity marks a likely topic change. Among the
gaps that keep a part within the size budget, the lowest-similarity one wins.
"""

import math
import re
from collections import Counter

WINDOW_LINES = 4          # lines compared on each side of a gap
MIN_FILL = 0.5            # a part is at least half the budget before we look for a cut

STOPWORDS = {
    "a", "about", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been",
    "but", "by", "can", "could", "did", "do", "does", "don", "for", "from", "get", "got", "had", "has",
    "have", "he", "her", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "know", "like", "ll", "m", "maybe", "me", "mean", "mm", "more", "my", "no", "not", "now", "of",
    "oh", "ok", "okay", "on", "one", "or", "our", "out", "re", "really", "right", "s", "say", "see",
    "she", "should", "so", "some", "something", "sort", "t", "that", "the", "their", "them", "then",
    "there", "these", "they", "thing", "things", "think", "this", "those", "to", "uh", "um", "up",
    "us", "ve", "very", "was", "we", "well", "were", "what", "when", "which", "who", "why", "will",
    "with", "would", "yeah", "yes", "you", "your",
}


def estimate_tokens(text: str) -> int:
    """Rough LLM token count (about 1.3 tokens per English word)."""
    return int(len(text.split()) * 1.3) + 1


def _words(text: str) -> Counter:
    return Counter(w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOPWORDS and len(w) > 2)


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[w] * b[w] for w in a.keys() & b.keys())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def gap_similarities(lines: list[str], window: int = WINDOW_LINES) -> list[float]:
    """sims[g] = similarity of the lines just before gap g with the lines just
    after it (gap g sits between line g-1 and line g). sims[0] is unused."""
    bags = [_words(t) for t in lines]
    sims = [1.0] * len(lines)
    for g in range(1, len(lines)):
        before = sum(bags[max(0, g - window):g], Counter())
        after = sum(bags[g:g + window], Counter())
        sims[g] = _cosine(before, after)
    return sims


def split_segments(lines: list[str], max_tokens: int) -> list[tuple[int, int]]:
    """Contiguous (start, end) line ranges covering all lines, each within
    max_tokens where possible, cut at the most likely topic change."""
    n = len(lines)
    tokens = [estimate_tokens(t) for t in lines]
    cum = [0]
    for t in tokens:
        cum.append(cum[-1] + t)
    if cum[n] <= max_tokens:
        return [(0, n)]

    sims = gap_similarities(lines)
    ranges, start = [], 0
    while start < n:
        if cum[n] - cum[start] <= max_tokens:
            ranges.append((start, n))
            break
        fits = [g for g in range(start + 1, n) if cum[g] - cum[start] <= max_tokens]
        candidates = [g for g in fits if cum[g] - cum[start] >= MIN_FILL * max_tokens]
        if candidates:
            cut = min(candidates, key=lambda g: (sims[g], -g))  # lowest similarity; later cut on ties
        else:
            cut = fits[-1] if fits else start + 1  # a single over-long line becomes its own part
        ranges.append((start, cut))
        start = cut
    return ranges
