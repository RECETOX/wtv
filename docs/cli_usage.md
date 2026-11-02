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
| `--rt_window` | Retention window for finding nearby compounds, in the units of `--retention_field`. | 2.0 (`retention_time`) / 20.0 (`retention_index`) |
| `--min_ion_intensity_percent` | Minimum ion intensity percentage. | 7 |
| `--min_ion_num` | Minimum number of ions. | 2 |
| `--prefer_mz_threshold` | Preferred m/z threshold. | 60 |
| `--similarity_threshold` | Similarity threshold. | 0.85 |
| `--fr_factor` | FR factor. | 2.0 |
| `--retention_time_max` | Maximum retention time or index value. | inf |
| `--retention_field` | Field to use for retention data; one of `retention_time` or `retention_index`. GC-MS libraries often provide a retention index instead of a retention time. | retention_time |

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

### Retention-Indexed Libraries

Some GC-MS libraries annotate compounds with a retention index instead of a
retention time. Select the field the input file provides:

```bash
wtv-cli \
    --msp_path library_with_ri.msp \
    --outpath ./output \
    --retention_field retention_index
```

`--rt_window` is interpreted in the units of `--retention_field`. If it is not
given, `2.0` is used for retention times (minutes) and `20.0` retention-index
units for retention indices (a warning is logged). Retention indices span
hundreds to thousands of units, so a minute-scale window carried over from
retention-time data would match almost no compounds as neighbours and the
neighbour-aware part of the algorithm would be silently skipped. `20.0` is a
starting point, not a validated optimum: check it against the index spread of
your own library. An explicit `--rt_window` always overrides the default.

If the selected field is absent from the file or holds no numeric value for any
spectrum, execution stops with an error message that names the retention field
actually found and the `--retention_field` value to rerun with, and the CLI
exits with code 1.

### Using with uv

If you installed with uv, run:

```bash
uv run wtv-cli --msp_path input.msp --outpath ./output --mz_min 35 --mz_max 400
```

## Output

The CLI generates an MSP file in the specified output directory containing the filtered spectra based on the ion selection algorithm.
