# WTV Documentation

Welcome to the WTV documentation. This project provides tools for ion selection in mass spectrometry data based on the WTV-2.0 algorithm.

## Features

- Command-line interface (CLI) for easy usage
- Library functions for programmatic access
- Support for MSP file format
- Configurable parameters for different use cases

## Quick Start

### Installation

```bash
uv sync --all-extras
```

### Running the CLI

```bash
uv run wtv-cli --msp_path input.msp --outpath ./output --mz_min 35 --mz_max 400
```

## Navigation

Explore the sections below:

- [Installation](installation.md) - How to install and set up the package
- [CLI Usage](cli_usage.md) - Using the command-line interface
- [Library Usage](library_usage.md) - Using the library programmatically
- [API Reference](api.md) - Detailed API documentation
  - [Ion Selection API](api_ion_selection.md)
  - [Utils API](api_utils.md)
  - [Similarity API](api_similarity.md)

## Acknowledgements

This project is based on the original work by Honglun Yuan et al., published in [WTV_2.0](https://doi.org/10.1016/j.molp.2024.04.012).
