from __future__ import annotations

import importlib.metadata

import zfit_pwa as m


def test_version():
    assert importlib.metadata.version("zfit_pwa") == m.__version__
