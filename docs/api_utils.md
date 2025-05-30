# API Reference: Utils

## `read_msp`

Reads an MSP file and returns metadata and retention time data.

### Parameters

- `msp_file_path` (str): Path to the MSP file.

### Returns

- `Tuple[Dict[str, Dict[int, int]], pd.DataFrame]`: Metadata and retention time data.

### Example

```python
from wtv.utils import read_msp

meta, rt_data = read_msp("input.msp")
```

## `write_msp`

Writes filtered ions to an MSP file.

### Parameters

- `ion_df` (pd.DataFrame): DataFrame containing ion data.
- `output_directory` (Path): Directory to save the output MSP file.
- `source_msp_file` (Path): Path to the source MSP file.

### Example

```python
from wtv.utils import write_msp

write_msp(ion_df, "output", "input.msp")
```
