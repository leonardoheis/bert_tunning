import re

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


def clean_text(text: str) -> str:
    text = text.replace("\f", " ").replace("\xa0", " ")
    text = re.sub(r"\|[-: ]+\|[-: |]+", "", text)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"#+ ", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def detect_foreign_municipality(
    text: str, known_municipality: str = Settings.OOD_TRAINED_MUNICIPALITY
) -> str | None:
    """Returns the name found in a "Municipalidad de <Name>" phrase when it doesn't match
    known_municipality; None if no such phrase appears at all, or every phrase found
    matches. A bare city-name substring search (e.g. for "Cordoba") is too loose -- it
    also matches street names ("Calle Cordoba") and unrelated mentions ("Universidad de
    Cordoba") that appear even in genuinely in-distribution training documents. Anchoring
    to the actual jurisdiction-claim phrase avoids that, at the cost of staying silent
    (None) for the ~11% of documents that never name their municipality this way at all --
    this is a deliberate independent signal, not folded into the OR-based in_distribution
    ensemble in src/inference/classify.py, since one boolean-ish fact ("this document
    names a different city") shouldn't be diluted by or blended with continuous z-scores."""
    for match in _MUNICIPALIDAD_DE_RE.findall(text):
        name = str(match)
        if not name.lower().startswith(known_municipality.lower()):
            return name
    return None
