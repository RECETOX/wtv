"""Shared fixtures for the wtv test suite."""

import re
from pathlib import Path

import pytest

from test_data import get_test_file

RETENTION_TIME_LINE = re.compile(r"^\s*(RETENTION_TIME|RETENTIONTIME|RT)\s*:", re.I)


@pytest.fixture
def msp_without_retention_time(tmp_path):
    """GC-MS test library with all retention time entries removed.

    Mirrors real GC-MS libraries that provide only a retention index, which is
    the input shape issue #25 is about.
    """
    source = Path(get_test_file("ei_spectra"))
    stripped = tmp_path / "ei_spectra_without_rt.msp"
    lines = [
        line
        for line in source.read_text().splitlines(keepends=True)
        if not RETENTION_TIME_LINE.match(line)
    ]
    assert not any(RETENTION_TIME_LINE.match(line) for line in lines)
    assert re.search(
        r"RETENTION_?INDEX", source.read_text(), re.I
    ), "test library must provide a retention index"
    stripped.write_text("".join(lines))
    return stripped
