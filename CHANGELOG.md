# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New `api_similarity.md` documentation for similarity functions
- Pre-commit hooks configuration with black, ruff, and isort
- **Support for GC datasets without retention time but with a retention index**
  - New CLI option `--retention_field` (`retention_time` or `retention_index`), and a matching `retention_field` parameter in `read_msp`, `load_data`, `parse_spectra`, and `run_ion_selection`
  - New `MissingRetentionDataError` exception, raised by `get_rt_data` when the selected retention field is absent from the metadata or holds no numeric value for any spectrum; the message names the alternative field and the option to rerun with
  - New `resolve_rt_window()` function and `RETENTION_FIELDS` / `RT_INDEX_WINDOW_DEFAULT` constants in `wtv/utils.py` for scale-aware retention windows

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
- `--rt_window` is now interpreted in the units of `--retention_field` and defaults to 20.0 retention-index units (with a warning) when `retention_index` is selected; 20.0 is a starting point to be checked against the index spread of the data, not a validated optimum. Behaviour for retention times is unchanged (CLI default 2.0, `run_ion_selection` default 1.0) and an explicitly passed window always takes precedence
- The CLI now catches `MissingRetentionDataError`, logs the message, and exits with code 1 instead of raising a traceback
- Updated documentation for retention-index support:
  - docs/cli_usage.md: documented `--retention_field`, corrected the `--retention_time_max` default from 68.80 to the actual default (infinity), documented `--rt_window` units and field-dependent defaults, added a retention-index example and the error-exit behaviour
  - docs/api_utils.md: corrected the `retention` parameter name to `retention_field` in `read_msp` and `parse_spectra`, documented `retention_field` for `load_data`, added `resolve_rt_window`, `MissingRetentionDataError`, `RT_INDEX_WINDOW_DEFAULT`, and `RETENTION_FIELDS`
  - docs/api_ion_selection.md: added the missing `retention_field` parameter row and documented the field-dependent `rt_window` default
  - docs/usage.md: retention parameters updated for retention-index data

### Removed
- Poetry-specific configuration from `pyproject.toml`
- unittest imports from test files

### Fixed
- String comparison bug in `wtv/utils.py` (`is not "NA"` changed to `!= "NA"`)
- Import ordering issues in `wtv/utils.py`
- Selecting a retention field the MSP file does not provide no longer crashes with a confusing `AttributeError: 'NoneType' object has no attribute 'set_index'`; it now raises `MissingRetentionDataError` with an actionable message

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
