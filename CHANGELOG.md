# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-03

### Added
- **GC libraries that carry only a retention index can now be processed**
  - `--retention_field`, and the matching `retention_field` parameter of `read_msp`, `load_data`, `parse_spectra` and `run_ion_selection`, have been there since v0.2.0. Selecting `retention_index` on a library without retention times stopped with an `AttributeError` (see Fixed), so in practice the option was usable only on libraries that also carry retention times. What is new in this release is the path behind it:
  - New `MissingRetentionDataError` exception, raised by `get_rt_data` when the selected retention field is absent from the metadata or holds no numeric value for any spectrum; the message names the alternative field and the option to rerun with
  - New `resolve_rt_window()` function and `RETENTION_FIELDS` / `RT_INDEX_WINDOW_DEFAULT` constants in `wtv/utils.py` for scale-aware retention windows
- **A documented way to bump the version**
  - `[tool.bumpversion]` in `pyproject.toml`: `uv run bump-my-version bump <part>` rewrites `[project].version` and the `uv.lock` entry for the root package together, taking the current version from the PEP 621 metadata so it is never declared in two places, and leaving commits and tags to the release branch (`bump-my-version` added to the `dev` extra; the lock file change is purely additive, no existing pin moved)
  - `wtv.__version__` and the `wtv-cli --version` option, both read from the installed distribution metadata rather than repeated in the source
  - `docs/releasing.md` with the version policy, the bump procedure, the release procedure and a checklist, linked from `README.md`, `docs/index.md` and the documentation navigation
  - `tests/test_version.py`, which pins the declared version, the installed metadata and the lock file to each other

### Changed
- `--rt_window` is now interpreted in the units of `--retention_field`. When it is not given, `RT_WINDOW_DEFAULT` (0.1 minutes) applies to retention times and `RT_INDEX_WINDOW_DEFAULT` (40 retention-index units, logged as a warning) to retention indices; `run_ion_selection` accepts `rt_window=None` for the same behaviour, and an explicitly passed window always takes precedence
- **The default retention window is no longer expressed in whole minutes.** The previous defaults (`--rt_window` 2.0 in the CLI, 1.0 in `run_ion_selection`) are far wider than the distance between co-eluting compounds on a GC gradient: `test_data/sample_data/chunk_0.msp` holds 87 compounds between 2.59 and 8.14 minutes at a median spacing of 0.04 minutes, so `rt_window=1.0` already returns 34.7 neighbours for the average compound and `rt_window=2.0` returns 61.4, out of 87 (each list including the compound itself) - the neighbour-aware selection degrades into comparing every compound against every other. At the new default the same figure is 4.4. The new defaults are calibrated to each other through the retention-index slope measured on the two test libraries that carry both fields (325-345 index units per minute), so 0.1 minutes and 40 index units describe the same physical window.
  - Reproducibility: the change alters results, so `test_data/ground_truth/chunk_0_filtered.msp` was regenerated and `tests/test_smoke.py` now pins an explicit, realistic parameter set instead of inheriting defaults. On `chunk_0.msp` the regenerated library differs from the previous one for 20 of 86 compounds and no longer contains `Tris(2-ethylhexyl) phosphate`; the total number of selected ions moves from 174 to 172.
  - Attributing that difference. The regeneration moved two things at once - the window, and the parameter set the smoke test now pins - so they were separated by running the v0.2.1 settings at the new window and m/z range: `mz_min=35, mz_max=400, rt_window=0.1` with the v0.2.1 defaults `min_ion_intensity_percent=5.0`, `min_ion_num=3` and `prefer_mz_threshold=150`. That intermediate library selects 183 ions and differs from the v0.2.1 library for 17 of the 86 compounds, so the window alone accounts for most of the change. The pinned parameters then account for 18 further differences against that intermediate run, including the dropped compound, which has its second strongest ion at 6 percent of the base peak and therefore offers fewer than the two ions the exclusion rule requires at any window size.
    - The three counts do not add up because they overlap, and here is the bookkeeping: 10 compounds differ in both steps, and 5 of those - Anthanthrene, Diazinone, Ipconazole, trans-Cinnamaldehyde, trans-Fenvalerate - end back at their v0.2.1 ion list, so 17 + 18 - 10 - 5 = 20.
    - Metric, chosen so the numbers above can be checked: per compound (keyed by `COMPOUND_NAME`) the ordered peak m/z values as read by matchms, compared between libraries. `min_ion_intensity_percent` moving from 5.0 to 7.0 is a change to the parameters the smoke test pins, not to a package default - the CLI default is 7.0 and `run_ion_selection` defaults to 5.0, at v0.2.1 and here alike. The intermediate library itself is not committed.
  - Limitation: neither default is validated against a benchmark. They are order-of-magnitude starting points, chosen to be comparable to the retention spread of the shipped test data; the window must still be checked against the gradient and peak widths of the library in use.
- The CLI now catches `MissingRetentionDataError`, logs the message, and exits with code 1 instead of raising a traceback
- Updated documentation for retention-index support:
  - docs/cli_usage.md: documented `--retention_field`, corrected the `--retention_time_max` default from 68.80 to the actual default (infinity), documented `--rt_window` units and field-dependent defaults, added a retention-index example and the error-exit behaviour
  - docs/api_utils.md: corrected the `retention` parameter name to `retention_field` in `read_msp` and `parse_spectra`, documented `retention_field` for `load_data`, added `resolve_rt_window`, `MissingRetentionDataError`, `RT_WINDOW_DEFAULT`, `RT_INDEX_WINDOW_DEFAULT`, and `RETENTION_FIELDS`, including the measured retention-index conversions of the two test libraries
  - docs/api_ion_selection.md and docs/library_usage.md: added the missing `retention_field` parameter row, documented the field-dependent `rt_window` default, and replaced example windows that were wide enough to compare the whole library
  - docs/usage.md: retention parameters updated for retention-index data
- Locked dependencies updated by Dependabot since v0.2.1: `certifi`; `virtualenv` with its `packaging` and `python-discovery` companions; and the uv group, i.e. `black`, `mkdocs-material` and `urllib3`. Nothing in `pyproject.toml` required a different version, so installing an older release still resolves as before

### Fixed
- Selecting a retention field the MSP file does not provide no longer crashes with a confusing `AttributeError: 'NoneType' object has no attribute 'set_index'`; it now raises `MissingRetentionDataError` with an actionable message

## [0.2.1] - 2026-07-10

Packaging only; the ion selection itself is untouched. Recorded retrospectively -
this release was tagged without updating this file, and the entries below sat in
`[Unreleased]` until v0.3.0.

### Changed
- The publish workflow uploads with `uv publish` instead of the `JRubics/poetry-publish` action, which no longer matched the uv/hatchling build
- The PyPI and bioconda badges in `README.md` now link to the packages they report

### Removed
- The internal planning notes `refactoring.md` and `tasks.md`

## [0.2.0] - 2026-07-10

Also recorded retrospectively: the migration below was carried in `[Unreleased]`
until v0.3.0, although it shipped with this release. Releases v0.0.1 (2025-05-19),
v0.0.2 (2025-05-30) and v0.1.0 (2025-07-03) predate this file and have no sections
of their own.

### Added
- Retention-data selection: the `--retention_field` CLI option and a matching `retention_field` parameter in `get_rt_data`, `read_msp`, `load_data`, `parse_spectra` and `run_ion_selection`. Only libraries carrying retention times worked with it - the retention-index path raised an `AttributeError` until v0.3.0
- `wtv/similarity/` as a package of its own, holding the weighted dot product and similarity scoring that v0.1.0 kept inside `wtv/ion_selection.py`, plus a matchms-based implementation
- `docs/api_similarity.md` documentation for similarity functions

The first version of this file listed an "initial release with ion selection
functionality", the CLI, MSP reading and writing, and the similarity calculations
under v0.2.0. Ion selection, the CLI and MSP reading and writing predate it -
v0.1.0 already shipped `wtv/cli.py`, the `wtv-cli` entry point and similarity
scoring - and the similarity calculations were only restructured here, not added.
The four bullets are dropped as an `Added` claim and reduced to this note, since
nothing in the repository records what v0.0.1 to v0.1.0 changed.

### Changed
- **Build system migrated from Poetry to uv**
  - Updated `pyproject.toml` to use PEP 621 standard
  - Changed build backend from `poetry-core` to `hatchling`
  - Dependencies now managed via `uv sync` instead of `poetry install`
- **Test system migrated from unittest to pytest**
  - Test files ported to pytest style
  - Tests use native Python asserts instead of unittest assertions
  - pytest fixtures used for test setup/teardown
  - Limitation: `tests/test_ion_selection.py` still groups its cases in a `unittest.TestCase` class, so the migration was not complete
- Updated documentation:
  - README.md updated with uv installation instructions
  - docs/installation.md updated with uv commands
  - docs/cli_usage.md enhanced with parameter tables
  - docs/library_usage.md expanded with examples
  - API documentation updated and reorganized
- Pre-commit configuration updated with black and isort hooks

### Removed
- Poetry-specific configuration from `pyproject.toml`

### Fixed
- String comparison bug in `wtv/utils.py` (`is not "NA"` changed to `!= "NA"`)
- Import ordering issues in `wtv/utils.py`

### Deprecated
- Poetry-based workflow (use `uv sync` instead)
- unittest-based tests (use pytest for new tests)

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
