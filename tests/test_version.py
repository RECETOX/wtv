"""The version must be declared once and stay reachable at runtime."""

import re
import sys
from pathlib import Path

import pytest

import wtv
from wtv import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_declared_version() -> str:
    """Return the version declared as ``[project].version``.

    ``tomllib`` would parse this more robustly, but it needs Python 3.11 and the
    package still supports 3.10.
    """
    lines = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    in_project = False
    for line in lines:
        if line.strip() == "[project]":
            in_project = True
        elif in_project and line.startswith("["):
            break
        elif in_project:
            match = re.fullmatch(r'\s*version = "([^"]+)"', line)
            if match:
                return match.group(1)
    raise AssertionError("pyproject.toml declares no [project].version")


def get_locked_version() -> str:
    """Return the version uv recorded for the root package in ``uv.lock``."""
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    recorded = re.search(
        r'^name = "wtv"\nversion = "([^"]+)"\nsource = \{ editable = "\." \}',
        lock,
        re.MULTILINE,
    )
    assert recorded is not None, 'uv.lock records no editable root package named "wtv"'
    return recorded.group(1)


class TestVersion:
    """The declared version must reach both the runtime and the lock file."""

    def test_runtime_version_matches_the_declaration(self):
        """``wtv.__version__`` must be what pyproject.toml declares."""
        assert wtv.__version__ == get_declared_version()

    def test_lock_file_matches_the_declaration(self):
        """A stale lock means environments were built from another version."""
        assert get_locked_version() == get_declared_version()

    def test_cli_reports_the_declared_version(self, monkeypatch, capsys):
        """``--version`` must print the declared version and start no work.

        The printed string has to be compared against the declaration, not against
        ``wtv.__version__``, which is what the CLI interpolates and would therefore
        match whatever wrong version both of them report.
        """
        calls = {}
        monkeypatch.setattr(
            cli, "run_ion_selection", lambda **kwargs: calls.update(kwargs)
        )
        monkeypatch.setattr(sys, "argv", ["wtv-cli", "--version"])

        with pytest.raises(SystemExit) as excinfo:
            cli.main()

        captured = capsys.readouterr()
        assert excinfo.value.code == 0
        assert captured.out.strip() == f"wtv-cli {get_declared_version()}"
        assert not calls
