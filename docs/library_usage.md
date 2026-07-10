# Library Usage

The WTV library provides programmatic access to its core functionality for ion selection in mass spectrometry data.

## Installation

First, install the library:

```bash
uv sync --all-extras
```

## Basic Example

Run ion selection using the `run_ion_selection` function:

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

## Core Functions

### `run_ion_selection`

Main function for performing ion selection on MSP files.

**Parameters:**
- `msp_file_path` (Path): Path to the input MSP file
- `output_directory` (Path): Directory to save output
- `mz_min` (float): Minimum m/z value (default: 0)
- `mz_max` (float): Maximum m/z value (default: infinity)
- `rt_window` (float): Retention time window for finding nearby compounds (default: 1)
- `min_ion_intensity_percent` (float): Minimum ion intensity percentage (default: 5)
- `min_ion_num` (int): Minimum number of ions required (default: 3)
- `prefer_mz_threshold` (float): Preferred m/z threshold (default: 150)
- `similarity_threshold` (float): Similarity threshold for compound matching (default: 0.85)
- `fr_factor` (float): Fragment ratio factor (default: 2)
- `retention_time_max` (float): Maximum retention time (default: infinity)

**Example:**

```python
from pathlib import Path
from wtv.ion_selection import run_ion_selection

# Run with custom parameters
run_ion_selection(
    msp_file_path=Path("data/sample.msp"),
    output_directory=Path("results"),
    mz_min=50,
    mz_max=500,
    rt_window=1.5,
    similarity_threshold=0.9,
)
```

### `generate_ion_combinations`

Generate optimal ion combinations for compounds.

**Parameters:**
- `min_ion_intensity_percent`: Minimum ion intensity percentage
- `min_ion_num`: Minimum number of ions
- `prefer_mz_threshold`: Preferred m/z threshold
- `similarity_threshold`: Similarity threshold
- `fr_factor`: Fragment ratio factor
- `RT_data`: DataFrame with retention time data
- `matrix`: DataFrame with ion intensities
- `rt_window`: Retention time window

**Example:**

```python
from wtv.utils import load_data
from wtv.ion_selection import generate_ion_combinations

# Load data
RT_data, matrix = load_data(Path("input.msp"), mz_min=35, mz_max=400)

# Generate ion combinations
combinations = generate_ion_combinations(
    min_ion_intensity_percent=7,
    min_ion_num=2,
    prefer_mz_threshold=60,
    similarity_threshold=0.85,
    fr_factor=2.0,
    RT_data=RT_data,
    matrix=matrix,
    rt_window=2.0,
)
```

## Utility Functions

### Data Loading

```python
from wtv.utils import load_data, read_msp, create_ion_matrix

# Load data from MSP file
RT_data, matrix = load_data(Path("input.msp"), mz_min=35, mz_max=400)

# Or use read_msp directly
ion_dict, rt_df = read_msp("input.msp")
```

### Writing Results

```python
from wtv.utils import write_msp

# Write filtered spectra to MSP file
write_msp(ion_rt_df, Path("output"), Path("input.msp"))
```

## Logging

The library uses Python's logging module. To see detailed logs:

```python
import logging

logging.basicConfig(level=logging.INFO)

from wtv.ion_selection import run_ion_selection
run_ion_selection(...)
```
