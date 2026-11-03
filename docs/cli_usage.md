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
| `--rt_window` | Retention window for finding nearby compounds, in the units of `--retention_field`. | 0.1 (`retention_time`) / 40.0 (`retention_index`) |
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
    --rt_window 0.1 \
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
    --rt_window 0.15 \
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
given, `0.1` is used for retention times (minutes) and `40.0` retention-index
units for retention indices (a warning is logged). The two numbers describe the
same physical window: the test libraries that carry both fields measure 325-345
index units per minute, so 40 index units are about 0.12 minutes.

Choosing the window is a matter of matching it to the distance between compounds
that genuinely co-elute, not to the length of the gradient. A window of whole
minutes on a fast GC gradient is the failure mode to avoid:
`test_data/sample_data/chunk_0.msp` holds 87 compounds between 2.59 and 8.14
minutes, a median of 0.04 minutes apart, so `--rt_window 1.0` returns 34.7
neighbours for the average compound and `--rt_window 2.0` returns 61.4, out of
87 - the selection degenerates into comparing every compound against every
other. At the 0.1 default the same figure is 4.4. Both defaults are starting points, not
validated optima: check them against the retention spread of your own library.
An explicit `--rt_window` always overrides the default.

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
