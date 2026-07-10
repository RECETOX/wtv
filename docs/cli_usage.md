# CLI Usage

The WTV CLI provides an easy way to perform ion selection on mass spectrometry data.

## Basic Command

```bash
wtv-cli --msp_path input.msp --outpath output --mz_min 35 --mz_max 400 ...
```

When using uv:

```bash
uv run wtv-cli --msp_path input.msp --outpath output ...
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--msp_path` | Path to the input MSP file. | Required |
| `--outpath` | Directory to save the output files. | Required |
| `--mz_min` | Minimum m/z value. | 35 |
| `--mz_max` | Maximum m/z value. | 400 |
| `--rt_window` | Retention time window. | 2.00 |
| `--min_ion_intensity_percent` | Minimum ion intensity percentage. | 7 |
| `--min_ion_num` | Minimum number of ions. | 2 |
| `--prefer_mz_threshold` | Preferred m/z threshold. | 60 |
| `--similarity_threshold` | Similarity threshold. | 0.85 |
| `--fr_factor` | FR factor. | 2.0 |
| `--retention_time_max` | Maximum retention time. | 68.80 |

## Examples

### Basic Usage

Run ion selection with default parameters:

```bash
wtv-cli \
    --msp_path input.msp \
    --outpath ./output \
    --mz_min 35 \
    --mz_max 400 \
    --rt_window 2.0 \
    --min_ion_intensity_percent 7 \
    --min_ion_num 2 \
    --prefer_mz_threshold 60 \
    --similarity_threshold 0.85 \
    --fr_factor 2.0 \
    --retention_time_max 68.80
```

### Custom Parameters

Adjust parameters for your specific data:

```bash
wtv-cli \
    --msp_path my_data.msp \
    --outpath ./results \
    --mz_min 50 \
    --mz_max 500 \
    --rt_window 1.5 \
    --similarity_threshold 0.9
```

### Using with uv

If you installed with uv, run:

```bash
uv run wtv-cli --msp_path input.msp --outpath ./output --mz_min 35 --mz_max 400
```

## Output

The CLI generates an MSP file in the specified output directory containing the filtered spectra based on the ion selection algorithm.
