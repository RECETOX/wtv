# Installation

## Using uv (Recommended)

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

This will:
- Create a virtual environment in `.venv`
- Install all dependencies including dev and docs extras
- Make the `wtv-cli` command available

### Common uv Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Sync dependencies (install packages) |
| `uv sync --all-extras` | Install all optional dependencies |
| `uv run <command>` | Run a command in the virtual environment |
| `uv add <package>` | Add a new dependency |
| `uv build` | Build distribution packages |

## Using pip

1. Install the library directly from PyPI:
   ```bash
   pip install wtv
   ```

2. For development, clone the repository and install in editable mode:
   ```bash
   git clone https://github.com/RECETOX/wtv.git
   cd wtv
   pip install -e ".[dev,docs]"
   ```

## Verify Installation

Run the following command to verify the installation:

```bash
wtv-cli --help
```

You should see the CLI help message with available options.
