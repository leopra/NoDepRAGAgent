from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-assets",
        action="store_true",
        help="When set, refresh the stored system prompt asset to match the runtime prompt.",
    )


@pytest.fixture
def update_assets(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-assets"))
