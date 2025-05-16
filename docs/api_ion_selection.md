# API Reference: Ion Selection

## `main_logic`
Core function for ion selection.

### Parameters
- `msp_file_path` (str): Path to the input MSP file.
- `output_directory` (str): Directory to save the output files.
- `mz_min` (int): Minimum m/z value.
- `mz_max` (int): Maximum m/z value.
- `rt_window` (float): Retention time window.
- `min_ion_intensity_percent` (float): Minimum ion intensity percentage.
- `min_ion_num` (int): Minimum number of ions.
- `prefer_mz_threshold` (int): Preferred m/z threshold.
- `similarity_threshold` (float): Similarity threshold.
- `fr_factor` (float): FR factor.
- `retention_time_max` (float): Maximum retention time.

### Example
```python
from src.ion_selection import main

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
