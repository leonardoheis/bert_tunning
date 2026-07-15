import re
from typing import NamedTuple

from src.settings import Settings

_MUNICIPALIDAD_DE_RE = re.compile(
    # [\s|]+ instead of \s+ throughout -- MarkItDown renders some PDF letterheads as
    # single-cell markdown tables, splitting "Municipalidad de la Ciudad de X" with a
    # stray "|" (e.g. "MUNICIPALIDAD DE LA | CIUDAD \n\nDE SANTA FE..."). Treating "|" as
    # just another separator, and excluding it from the captured name itself, stops that
    # bare pipe from being captured as the "name" in place of the real one that follows.
    r"municipalidad[\s|]+de[\s|]+(?:la[\s|]+)?(?:ciudad[\s|]+de[\s|]+)?([^\s,.;:()|\n]+)",
    re.IGNORECASE,
)
# Chars of raw text kept on each side of a match for ForeignMunicipalityMatch.context --
# wide enough to usually cover a multi-word name ("Santa Fe de la Vera Cruz") without
# guessing where it ends the way a wider capture group would have to.
_CONTEXT_CHARS = 40


class ForeignMunicipalityMatch(NamedTuple):
    """detect_foreign_municipality's result. `name` is only the single word immediately
    after "Municipalidad de" -- reliable for the known-municipality comparison since it's
    anchored to the literal phrase, but too little on its own to confirm by eye that it's
    really part of a longer name ("Santa" vs. "Santa Fe de la Vera Cruz") rather than
    something else -- silently guessing where a multi-word name ends risks swallowing
    unrelated trailing text on sentences with no punctuation (this corpus's ALL-CAPS
    letterheads often have none). `context` sidesteps that by carrying the surrounding raw
    text instead of a guessed word count, so the match can be verified against the source."""

    name: str
    context: str


def clean_text(text: str) -> str:
    text = text.replace("\f", " ").replace("\xa0", " ")
    text = re.sub(r"\|[-: ]+\|[-: |]+", "", text)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"#+ ", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def detect_foreign_municipality(
    text: str, known_municipality: str | None = None
) -> ForeignMunicipalityMatch | None:
    """Returns a ForeignMunicipalityMatch for the first "Municipalidad de <Name>" phrase
    found whose Name doesn't match known_municipality; None if no such phrase appears at
    all, or every phrase found matches. A bare city-name substring search (e.g. for
    "Cordoba") is too loose -- it also matches street names ("Calle Cordoba") and
    unrelated mentions ("Universidad de Cordoba") that appear even in genuinely
    in-distribution training documents. Anchoring to the actual jurisdiction-claim phrase
    avoids that, at the cost of staying silent (None) for the ~11% of documents that never
    name their municipality this way at all -- this is a deliberate independent signal,
    not folded into the OR-based in_distribution ensemble in src/inference/classify.py,
    since one boolean-ish fact ("this document names a different city") shouldn't be
    diluted by or blended with continuous z-scores. known_municipality defaults to None,
    read from Settings inside the body rather than as a function-default expression --
    Settings is a module-level singleton constructed once at import, so a
    `= Settings.OOD_TRAINED_MUNICIPALITY` default would freeze to whatever that held at
    import time instead of reflecting later reads."""
    reference = known_municipality or Settings.OOD_TRAINED_MUNICIPALITY
    for match in _MUNICIPALIDAD_DE_RE.finditer(text):
        name = match.group(1)
        if not name.lower().startswith(reference.lower()):
            start = max(0, match.start() - _CONTEXT_CHARS)
            end = min(len(text), match.end() + _CONTEXT_CHARS)
            context = " ".join(text[start:end].split())
            return ForeignMunicipalityMatch(name=name, context=context)
    return None
