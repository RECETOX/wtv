# API Reference

This document provides an overview of the main modules and functions in the wtv library.

## Modules

- [`wtv.utils`](api_utils.md) - Utility functions for MSP file I/O and data processing
- [`wtv.ion_selection`](api_ion_selection.md) - Core ion selection algorithms
- [`wtv.similarity`](api_similarity.md) - Similarity calculation functions

## Quick Start

```python
from pathlib import Path
from wtv.ion_selection import run_ion_selection

run_ion_selection(
    msp_file_path=Path("input.msp"),
    output_directory=Path("output"),
)
```
