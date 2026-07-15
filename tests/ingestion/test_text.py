from src.ingestion._text import detect_foreign_municipality


def test_detect_foreign_municipality_returns_none_when_matches_known() -> None:
    text = "LA MUNICIPALIDAD DE ROSARIO HA SANCIONADO LA SIGUIENTE ORDENANZA"
    assert detect_foreign_municipality(text, known_municipality="rosario") is None


def test_detect_foreign_municipality_returns_none_when_phrase_absent() -> None:
    text = "Esta Municipalidad dicta la presente ordenanza sin nombrarse."
    assert detect_foreign_municipality(text, known_municipality="rosario") is None


def test_detect_foreign_municipality_returns_name_when_different() -> None:
    text = "veh�culos de prensa acreditados ante la Municipalidad de Cordoba"
    match = detect_foreign_municipality(text, known_municipality="rosario")
    assert match is not None
    assert match.name == "Cordoba"


def test_detect_foreign_municipality_returns_context_around_match() -> None:
    text = "veh�culos de prensa acreditados ante la Municipalidad de Cordoba, anualmente"
    match = detect_foreign_municipality(text, known_municipality="rosario")
    assert match is not None
    assert "Municipalidad de Cordoba" in match.context


def test_detect_foreign_municipality_ignores_street_name_mention() -> None:
    # "Calle Cordoba" / "Universidad de Cordoba" are not jurisdiction claims -- a bare
    # substring search on the city name would false-positive on these, anchoring to the
    # "Municipalidad de" phrase avoids it.
    text = "DOMICILIO: Cordoba 721 - 10 Piso - ROSARIO. Estudi� en la Universidad de Cordoba."
    assert detect_foreign_municipality(text, known_municipality="rosario") is None


def test_detect_foreign_municipality_handles_ciudad_de_prefix() -> None:
    text = "esta Municipalidad de la Ciudad de Santa Fe informa que"
    match = detect_foreign_municipality(text, known_municipality="rosario")
    assert match is not None
    assert match.name == "Santa"


def test_detect_foreign_municipality_skips_stray_table_pipe() -> None:
    # Real MarkItDown output for a letterhead rendered as a single-cell markdown table --
    # the bare "|" must not itself be captured as the municipality name.
    text = 'MUNICIPALIDAD DE LA | CIUDAD \n\nDE SANTA FE DE LA VERA CRUZ "202(J -A�TOdel GC�l�a'
    match = detect_foreign_municipality(text, known_municipality="rosario")
    assert match is not None
    assert match.name == "SANTA"
    # Context must still carry the rest of the name the truncated `name` field can't --
    # this is the whole point of returning it alongside the single-word anchor.
    assert "VERA CRUZ" in match.context
