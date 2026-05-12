"""Smoke tests — verify nvidia_nemotron_model_reasoning_challenge imports."""


def test_import_package():
    import nvidia_nemotron_model_reasoning_challenge as pkg

    assert pkg.__version__
