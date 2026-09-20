"""Shared pytest fixtures."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from clusterfed.config import load_config, set_global_seed  # noqa: E402
from synthetic import make_frame  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    c = load_config(config_path)
    set_global_seed(int(c["seed"]))
    return c


@pytest.fixture()
def raw_frame():
    return make_frame(seed=0)
