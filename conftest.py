import sys
from pathlib import Path

# `tests/` holds helper modules (the stub API) that the tests import directly.
sys.path.insert(0, str(Path(__file__).parent / "tests"))
