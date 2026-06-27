def test_src_package_importable() -> None:
    import src  # noqa: PLC0415

    assert src is not None


def test_ingestion_package_importable() -> None:
    import src.ingestion  # noqa: PLC0415

    assert src.ingestion is not None


def test_training_package_importable() -> None:
    import src.training  # noqa: PLC0415

    assert src.training is not None


def test_inference_package_importable() -> None:
    import src.inference  # noqa: PLC0415

    assert src.inference is not None


def test_api_package_importable() -> None:
    import src.api  # noqa: PLC0415

    assert src.api is not None


def test_cli_package_importable() -> None:
    import src.cli  # noqa: PLC0415

    assert src.cli is not None
