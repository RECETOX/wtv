# Usage

This document provides quick reference for using WTV.

## CLI Usage

```bash
uv run wtv-cli --msp_path input.msp --outpath output --mz_min 35 --mz_max 400
```

See [CLI Usage](cli_usage.md) for detailed documentation.

## Library Usage

```python
from pathlib import Path
from wtv.ion_selection import run_ion_selection

run_ion_selection(
    msp_file_path=Path("input.msp"),
    output_directory=Path("output"),
    mz_min=35,
    mz_max=400,
)
```

See [Library Usage](library_usage.md) for detailed documentation.

## Common Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `mz_min` | Minimum m/z value | 35-100 |
| `mz_max` | Maximum m/z value | 400-1000 |
| `rt_window` | Retention time window | 1.0-3.0 |
| `similarity_threshold` | Similarity threshold | 0.8-0.95 |
| `min_ion_num` | Minimum ions required | 2-5 |
