import pytest


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Keep the repo's config.toml out of unit tests; each test builds its own Settings."""
    monkeypatch.setenv("GPTE_CONFIG", str(tmp_path / "absent.toml"))
