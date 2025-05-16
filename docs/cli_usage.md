# CLI Usage

The WTV CLI provides an easy way to perform ion selection.

## Basic Command
```bash
wtv-cli --msp_path input.msp --outpath output --mz_min 35 --mz_max 400 ...
```

## Options
- `--msp_path`: Path to the input MSP file.
- `--outpath`: Directory to save the output files.
- `--mz_min`: Minimum m/z value (default: 35).
- `--mz_max`: Maximum m/z value (default: 400).
- `--rt_window`: Retention time window (default: 2.00).
- `--min_ion_intensity_percent`: Minimum ion intensity percentage (default: 7).
- `--min_ion_num`: Minimum number of ions (default: 2).
- `--prefer_mz_threshold`: Preferred m/z threshold (default: 60).
- `--similarity_threshold`: Similarity threshold (default: 0.85).
- `--fr_factor`: FR factor (default: 2.0).
- `--retention_time_max`: Maximum retention time (default: 68.80).

## Example
```bash
wtv-cli --msp_path input.msp --outpath output --mz_min 50 --mz_max 500
```
