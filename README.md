![PyPI - Version](https://img.shields.io/pypi/v/wtv)
![Conda Version](https://img.shields.io/conda/vn/bioconda/wtv?style=flat)


# wtv

An implementation of ion selection based on WTV-2.0

## Project Setup

### Using uv (Recommended)

1. **Install uv** if you haven't already:
   ```bash
   # Using pip
   pip install uv

   # Or using Homebrew on macOS
   brew install uv

   # Or using curl
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/RECETOX/wtv.git
   cd wtv
   ```

3. **Sync dependencies and create virtual environment**:
   ```bash
   uv sync --all-extras
   ```

4. **Run commands within the virtual environment**:
   ```bash
   uv run wtv-cli --help
   ```

### Using pip

1. Install the library directly from PyPI:
   ```bash
   pip install wtv
   ```

## Running Tests

### Using pytest

1. Ensure you have the development dependencies installed:
   ```bash
   uv sync --all-extras
   ```

2. Run tests with pytest:
   ```bash
   uv run pytest
   ```

3. Run tests with verbose output:
   ```bash
   uv run pytest -v
   ```

4. Generate a test coverage report:
   ```bash
   uv run pytest --cov=wtv
   ```

## Building the Package

To build distribution packages (wheel and source distribution):

```bash
uv build
```

This will create:
- `dist/wtv-<version>-py3-none-any.whl` - Wheel package
- `dist/wtv-<version>.tar.gz` - Source distribution

Verify the installation after building:

```bash
pip install dist/wtv-*.whl
wtv-cli --help
```

## Documentation

### Testing Documentation Locally

To test the documentation locally:

1. **Serve the documentation**:
   ```bash
   uv run mkdocs serve
   ```

2. **Access the documentation**:
   Open your browser and navigate to `http://127.0.0.1:8000` to view the documentation.

See the [docs](docs/) folder for more information.

### Documentation Deployment

This project uses GitHub Actions to auto-generate and deploy documentation to GitHub Pages.

## Running GitHub Actions Locally with `act`

To run all GitHub Actions workflows locally using `act`:

1. **Install act**:
   Download and install `act` from its [GitHub repository](https://github.com/nektos/act).

2. **Set Up Secrets**:
   Create a `.secrets` file in the root of your repository and define the required secrets:
   ```
   PYPI_API_TOKEN=your-real-or-mock-token-for-testing
   ```

3. **Run All Workflows**:
   ```bash
   act
   ```

4. **Run a Specific Workflow**:
   ```bash
   act -W .github/workflows/package.yaml
   act -W .github/workflows/publish.yml
   ```

5. **Simulate Events**:
   ```bash
   act -e push
   ```

6. **Use a Specific Runner**:
   ```bash
   act -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-latest
   ```

## Known Issues

- It seems to generate slightly different results based on the OS version and/or Python version.

## Acknowledgements

This project is based on the original work by Honglun Yuan, Yiding Jiangfang, Zhenhua Liu, Rong Rong Su, Qiao Li, Chuanying Fang, Sishu Huang, Xianqing Liu, Alisdair Robert Fernie, and Jie Luo, as published in [WTV_2.0](https://github.com/yuanhonglun/WTV_2.0) and [their associated publication](https://doi.org/10.1016/j.molp.2024.04.012).
