"""Select lane source and a task-owned Matplotlib cache; never historical outputs."""
import atexit
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_cache = tempfile.TemporaryDirectory(prefix="pymcp-test-mpl-")
atexit.register(_cache.cleanup)
os.environ.setdefault("MPLCONFIGDIR", _cache.name)

import matplotlib

matplotlib.use("Agg")
