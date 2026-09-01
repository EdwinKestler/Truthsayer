"""Lightweight linguistic markers (no BERT).

Newman et al. (2003) and later reviews find *small* lexical effects
(fewer first-person pronouns, more negative emotion in some tasks).
These are not a lie classifier. Used only when a transcript is supplied.
"""
from __future__ import annotations

import re

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our", "ours"}
HEDGES = {
    "maybe", "perhaps", "possibly", "probably", "sort", "kinda", "kind",
    "guess", "think", "believe", "seem", "seems", "somewhat", "around",
}
NEGATIONS = {"no", "not", "never", "n't", "none", "nobody", "nothing"}
FILLERS = {"um", "uh", "er", "ah", "like", "youknow"}
NEG_AFFECT = {
    "afraid", "angry", "bad", "hate", "hurt", "nervous", "scared",
    "sorry", "terrible", "upset", "worried",
}


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


class LinguisticCues:
    def __init__(self):
        self.last = {}

    def update(self, text):
        tokens = tokenize(text)
        n = max(len(tokens), 1)
        counts = {
            "first_person": sum(1 for t in tokens if t in FIRST_PERSON),
            "hedges": sum(1 for t in tokens if t in HEDGES),
            "negations": sum(1 for t in tokens if t in NEGATIONS or t.endswith("n't")),
            "fillers": sum(1 for t in tokens if t.replace("'", "") in FILLERS),
            "neg_affect": sum(1 for t in tokens if t in NEG_AFFECT),
        }
        rates = {k: v / n for k, v in counts.items()}
        flags = {
            "low_first_person": rates["first_person"] < 0.03 and n >= 12,
            "high_hedging": rates["hedges"] > 0.06,
            "high_negation": rates["negations"] > 0.08,
            "high_filler": rates["fillers"] > 0.08,
            "neg_affect": rates["neg_affect"] > 0.04,
        }
        self.last = {
            "n_tokens": n,
            "rates": rates,
            "flags": {k: v for k, v in flags.items() if v},
            "text_excerpt": (text or "")[:240],
        }
        return self.last

    def snapshot(self):
        return dict(self.last)
