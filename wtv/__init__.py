# This file makes the `wtv` directory a package.
#
# The version is taken from the installed distribution metadata, which the build
# backend reads from `[project].version` in `pyproject.toml`. That keeps the
# version declared in exactly one place. See docs/releasing.md.
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wtv")
except PackageNotFoundError:
    # A source tree that was never installed. Importing the package must not fail
    # over it; "0+unknown" is a legal PEP 440 version that no release can have.
    __version__ = "0+unknown"

__all__ = ["__version__"]
