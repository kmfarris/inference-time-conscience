"""
tokenizer.py — Shared tokenization for Scatterbrain

Provides a simple stemmer that normalizes word forms so that
"destroying", "destroyed", "destroys", and "destroy" all map to
the same concept. This improves SNN concept matching without
requiring an embedding model or subword tokenizer.

The stemmer is intentionally simple — suffix stripping only.
No lemmatization, no POS tagging. This keeps it fast enough
for real-time inference and portable to Pi Zero.
"""

import re


# Suffixes to strip, ordered by length (longest first)
_SUFFIXES = [
    "ingly", "edly", "ingly", "ously", "iously",
    "ing", "ed", "ies", "ied", "ies", "ied",
    "ous", "ful", "less", "ness", "ment", "tion", "sion",
    "able", "ible", "ally", "ingly",
    "ly", "es", "er", "est", "or",
    "s",
]

# Words that should NOT be stemmed (stemming would corrupt them)
_KEEP_AS_IS = {
    "is", "was", "has", "his", "its", "us", "as", "ss",  # "ss" from "boss"
    "this", "thus", "plus", "us", "gas", "bus",
    "yes", "no", "not", "nor", "for", "but", "yet",
    "compassion", "creation",  # core values — keep intact
    # Negation words — must not be stemmed or negation detection breaks
    "never", "without", "cannot", "cannot", "cant", "dont",
    "wont", "shouldnt", "mustnt", "unable",
}


def simple_stem(word: str) -> str:
    """
    Simple suffix-stripping stemmer.

    "destroying" -> "destroy"
    "fostered" -> "foster"
    "wisdom" -> "wisdom" (no suffix to strip)
    "compassion" -> "compassion" (protected)
    """
    if word in _KEEP_AS_IS:
        return word

    if len(word) < 4:
        return word

    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]

    return word


def clean_tokenize(text: str, stem: bool = True) -> list:
    """
    Tokenize and optionally stem text.

    Splits on all non-alpha characters, filters garbage tokens,
    and optionally applies simple stemming.
    """
    cleaned = re.sub(r'[^a-z\s]', ' ', text.lower())
    tokens = cleaned.split()

    result = []
    for t in tokens:
        if len(t) > 20:
            continue
        if len(t) < 2 and t not in ('i', 'a'):
            continue
        if stem:
            result.append(simple_stem(t))
        else:
            result.append(t)

    return result