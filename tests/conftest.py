"""Let pure-logic tests import modules that import pysam or pyBigWig where those can't be installed.

pysam has no Windows build. When it (or pyBigWig) is missing, a stand-in module is registered
whose attributes are placeholder classes, so imports and type annotations such as
`pysam.AlignedSegment` resolve; tests that need the real library are marked
`requires_pysam` and skipped. On the server, where both are installed, nothing is stubbed.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

for _name in ("pysam", "pyBigWig"):
    try:
        importlib.import_module(_name)
    except ImportError:
        _stub = types.ModuleType(_name)
        _stub.__getattr__ = lambda attr: type(attr, (), {})
        _stub.IS_TEST_STUB = True
        sys.modules[_name] = _stub

# Checked on the module itself: this file may be imported a second time (as tests.conftest),
# when pysam already resolves to the stub.
requires_pysam = pytest.mark.skipif(
    vars(sys.modules["pysam"]).get("IS_TEST_STUB", False), reason="pysam is not installed"
)
