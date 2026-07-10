# API Reference: Ion Selection

This document describes the ion selection functions in `wtv.ion_selection`.

## `run_ion_selection`

Main function for performing ion selection on MSP files.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `msp_file_path` | Path | Required | Path to the input MSP file |
| `output_directory` | Path | Required | Directory to save the output files |
| `mz_min` | float | 0 | Minimum m/z value |
| `mz_max` | float | inf | Maximum m/z value |
| `rt_window` | float | 1 | Retention time window for finding nearby compounds |
| `min_ion_intensity_percent` | float | 5 | Minimum ion intensity percentage |
| `min_ion_num` | int | 3 | Minimum number of ions required |
| `prefer_mz_threshold` | float | 150 | Preferred m/z threshold |
| `similarity_threshold` | float | 0.85 | Similarity threshold for compound matching |
| `fr_factor` | float | 2 | Fragment ratio factor |
| `retention_time_max` | float | inf | Maximum retention time |

### Example

```python
from pathlib import Path
from wtv.ion_selection import run_ion_selection

run_ion_selection(
    msp_file_path=Path("input.msp"),
    output_directory=Path("output"),
    mz_min=35,
    mz_max=400,
    rt_window=2.00,
    min_ion_intensity_percent=7,
    min_ion_num=2,
    prefer_mz_threshold=60,
    similarity_threshold=0.85,
    fr_factor=2.0,
    retention_time_max=68.80,
)
```

## `generate_ion_combinations`

Generate optimal ion combinations for compounds based on similarity and retention time proximity.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_ion_intensity_percent` | float | Minimum ion intensity percentage |
| `min_ion_num` | int | Minimum number of ions |
| `prefer_mz_threshold` | float | Preferred m/z threshold |
| `similarity_threshold` | float | Similarity threshold |
| `fr_factor` | float | Fragment ratio factor |
| `RT_data` | pd.DataFrame | DataFrame with retention time data |
| `matrix` | pd.DataFrame | DataFrame with ion intensities |
| `rt_window` | float | Retention time window |

### Returns

- `pd.DataFrame`: DataFrame containing ion combinations for each compound

## Other Functions

### `get_ion_rt`

Collect retention times for selected ions.

### `get_nearby_compounds`

Find compounds within a retention time window.

### `filter_matrix`

Filter ions by intensity threshold.

### `calculate_ion_combination`

Calculate optimal ion combination for a compound.
