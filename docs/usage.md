# Usage

## CLI Usage

```bash
wtv-cli --msp_path input.msp --outpath output --mz_min 35 --mz_max 400 ...
```

## Library Usage

```python
from wtv.ion_selection import main_logic

main_logic(
    msp_file_path="input.msp",
    output_directory="output",
    mz_min=35,
    mz_max=400,
    ...
)
```
