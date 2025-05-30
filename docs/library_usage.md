# Library Usage

The WTV library provides programmatic access to its core functionality.

## Example Usage

```python
from wtv.ion_selection import main

main(
    msp_file_path="input.msp",
    output_directory="output",
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

## Functions

- `main`: Core function for ion selection.
- `read_msp`: Reads an MSP file and returns metadata and retention time data.
- `write_msp`: Writes filtered ions to an MSP file.
