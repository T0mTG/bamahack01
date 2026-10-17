import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "controllers" / "rl_longitudinal"))


@pytest.fixture
def cfg():
    from autocross.config import load_sim_config
    c = load_sim_config()
    c["backend"] = "bicycle"
    c["output"]["plot"] = False
    return c


@pytest.fixture
def root():
    return ROOT
