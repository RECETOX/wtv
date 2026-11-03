# API Reference: Utils

This document describes the utility functions in `wtv.utils`.

## `read_msp`

Reads an MSP file and returns metadata and retention time data using matchms.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `msp_file_path` | str or Path | Required | Path to the MSP file |
| `retention_field` | str | "retention_time" | Name of the retention field: "retention_time" or "retention_index" |

### Returns

- `Tuple[Dict[str, Dict[float, int]], pd.DataFrame]`:
  - Dictionary of compound names to ion intensities
  - DataFrame with columns 'Name' and 'RT'

### Example

```python
from wtv.utils import read_msp

meta, rt_data = read_msp("input.msp")
print(meta.keys())  # Compound names
print(rt_data)      # Retention times
```

## `write_msp`

Writes filtered ions to an MSP file.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `ion_df` | pd.DataFrame | DataFrame containing ion data with 'RT' and 'ion' columns |
| `output_directory` | Path | Directory to save the output MSP file |
| `source_msp_file` | Path | Path to the source MSP file |

### Example

```python
from pathlib import Path
from wtv.utils import write_msp

write_msp(ion_df, Path("output"), Path("input.msp"))
```

## `load_data`

Load and prepare data from an MSP file for ion selection.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `msp_file_path` | Path | Path to the MSP file |
| `mz_min` | float | Minimum m/z value |
| `mz_max` | float | Maximum m/z value |
| `retention_field` | str | Name of the retention field: "retention_time" or "retention_index" (default: "retention_time") |

### Returns

- `Tuple[pd.DataFrame, pd.DataFrame]`: RT_data and ion matrix

## `create_ion_matrix`

Create a matrix of ion intensities with compounds as rows and m/z values as columns.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `mz_min` | float | Minimum m/z value |
| `mz_max` | float | Maximum m/z value |
| `meta_1` | dict | Dictionary of compound ion data |

### Returns

- `pd.DataFrame`: Ion intensity matrix

## `parse_spectra`

Parse spectra to create a matrix of ion intensities and a DataFrame of retention times.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spectra` | list[Spectrum] | Required | List of matchms Spectrum objects |
| `retention_field` | str | "retention_time" | Name of the retention field: "retention_time" or "retention_index" |
| `mz_min` | float | 0 | Minimum m/z value |
| `mz_max` | float | 1000 | Maximum m/z value |

### Returns

- `Tuple[pd.DataFrame, pd.DataFrame]`: Ion intensity matrix and RT DataFrame

## `resolve_rt_window`

Select the retention window when the caller left it unset. An explicitly
supplied window is returned unchanged; otherwise a default is chosen. Because
retention times and retention indices live on different scales there is no
single scale-free default: for `retention_field="retention_index"` the function
returns `RT_INDEX_WINDOW_DEFAULT` and logs a warning asking the user to review
the value against the index spread of their data. A substituted retention-time
default is logged at `INFO` level, because a window that is orders of magnitude
too wide is invisible in the output otherwise.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rt_window` | float or None | Required | User supplied window; `None` selects a default |
| `retention_field` | str | Required | Field used for retention data |

### Returns

- `float`: The window to apply, in the units of `retention_field`

### Example

```python
from wtv.utils import resolve_rt_window

resolve_rt_window(None, "retention_index")  # -> 40.0 (logs a warning)
resolve_rt_window(None, "retention_time")   # -> 0.1
resolve_rt_window(1.5, "retention_index")   # -> 1.5
```

## `MissingRetentionDataError`

Exception (subclass of `ValueError`) raised by `get_rt_data` - and therefore by
`read_msp`, `load_data`, and `parse_spectra` - when the selected retention field
is absent from the spectrum metadata or holds no numeric value for any spectrum.
The message names the retention field that was found instead, when one exists,
and states the option to rerun with (for example
`--retention_field retention_index`). The CLI catches this error, logs the
message, and exits with code 1 instead of raising a traceback.

## `RT_WINDOW_DEFAULT`

Module constant, `0.1`. Default retention window in minutes, used by
`resolve_rt_window` when `retention_field="retention_time"` and no explicit
`rt_window` was given. It replaces the previous default of one minute, which on
a fast GC gradient spans tens of compounds and turns the neighbour-aware
selection into an all-compared-against-all search.

## `RT_INDEX_WINDOW_DEFAULT`

Module constant, `40.0`. Default retention window in retention-index units, used
by `resolve_rt_window` when `retention_field="retention_index"` and no explicit
`rt_window` was given. Retention indices span hundreds to thousands of units, so
neither a minute-scale value nor the retention-time default can be reused
numerically.

The two constants are calibrated against each other through the gradient slope.
The test libraries that carry both fields give the conversion directly:

| File | Conversion (least squares fit) | Index units per minute |
|------|-------------------------------|------------------------|
| `test_data/sample_data/chunk_0.msp` | RI = 345.09 · RT + 269.80 | 345 |
| `test_data/sample_data/ei_spectra.msp` | RI = 324.95 · RT + 355.50 | 325 |

At that slope, 40 index units are 0.12 minutes - the same physical window as
`RT_WINDOW_DEFAULT`. Both values are starting points; tune them to the retention
spread, gradient, and peak widths of your own data.

## `RETENTION_FIELDS`

Module constant, `("retention_time", "retention_index")`: the retention metadata
fields understood by the package. These are the valid values for the
`retention_field` parameters and for the CLI `--retention_field` option.

## `average_rts_for_duplicated_indices`

Average the retention times for duplicated compound indices.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `RT_data` | pd.DataFrame | DataFrame with retention time data |

### Returns

- `pd.DataFrame`: DataFrame with averaged retention times

## `get_filtered_spectra`

Generate filtered spectra based on ion combinations.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `spectra` | list[Spectrum] | List of Spectrum objects |
| `combinations` | pd.DataFrame | DataFrame with ion combinations |

### Returns

- `Generator[Spectrum, None, None]`: Generator of filtered spectra
