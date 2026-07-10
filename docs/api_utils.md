# API Reference: Utils

This document describes the utility functions in `wtv.utils`.

## `read_msp`

Reads an MSP file and returns metadata and retention time data using matchms.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `msp_file_path` | str or Path | Required | Path to the MSP file |
| `retention` | str | "retention_time" | Name of the retention field |

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
| `retention` | str | "retention_time" | Name of the retention field |
| `mz_min` | float | 0 | Minimum m/z value |
| `mz_max` | float | 1000 | Maximum m/z value |

### Returns

- `Tuple[pd.DataFrame, pd.DataFrame]`: Ion intensity matrix and RT DataFrame

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
