# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New `api_similarity.md` documentation for similarity functions
- Pre-commit hooks configuration with black, ruff, and isort

### Changed
- **Build system migrated from Poetry to uv**
  - Updated `pyproject.toml` to use PEP 621 standard
  - Changed build backend from `poetry-core` to `hatchling`
  - Dependencies now managed via `uv sync` instead of `poetry install`
- **Test system migrated from unittest to pytest**
  - All test files ported to pytest style
  - Tests use native Python asserts instead of unittest assertions
  - pytest fixtures used for test setup/teardown
- Updated documentation:
  - README.md updated with uv installation instructions
  - docs/installation.md updated with uv commands
  - docs/cli_usage.md enhanced with parameter tables
  - docs/library_usage.md expanded with examples
  - API documentation updated and reorganized
- Pre-commit configuration updated with black and isort hooks

### Removed
- Poetry-specific configuration from `pyproject.toml`
- unittest imports from test files

### Fixed
- String comparison bug in `wtv/utils.py` (`is not "NA"` changed to `!= "NA"`)
- Import ordering issues in `wtv/utils.py`

### Deprecated
- Poetry-based workflow (use `uv sync` instead)
- unittest-based tests (all tests now use pytest)

## [0.2.0] - Previous Release

### Added
- Initial release with ion selection functionality
- CLI interface
- MSP file reading and writing
- Similarity calculations for compound matching

---

## Migration Guide

### From Poetry to uv

If you were using Poetry to manage this project, here's how to migrate:

1. **Install uv**:
   ```bash
   pip install uv
   ```

2. **Remove Poetry environment**:
   ```bash
   poetry env remove
   ```

3. **Sync with uv**:
   ```bash
   uv sync --all-extras
   ```

4. **Run commands**:
   ```bash
   # Instead of: poetry run wtv-cli --help
   uv run wtv-cli --help

   # Instead of: poetry run pytest
   uv run pytest
   ```

### Command Reference

| Old (Poetry) | New (uv) |
|--------------|----------|
| `poetry install` | `uv sync` |
| `poetry install --all-extras` | `uv sync --all-extras` |
| `poetry add <package>` | `uv add <package>` |
| `poetry run <command>` | `uv run <command>` |
| `poetry build` | `uv build` |
