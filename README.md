![PyPI - Version](https://img.shields.io/pypi/v/wtv)
![Conda Version](https://img.shields.io/conda/vn/bioconda/wtv?style=flat)


# wtv

An implementation of ion selection based on WTV-2.0

### Project setup:

1: Clone the code from [wtv](https://github.com/RECETOX/wtv)

2: Create an empty miniconda environment [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

3: Manage dependencies with [poetry](https://python-poetry.org/)

### Running tests:

#### Using `unittest`:

To run all tests using `unittest`, use the following command:

```bash
python -m unittest discover -s tests
```

#### Using `pytest`:

1. Ensure `pytest` is installed:

   ```bash
   poetry install --with dev
   ```

2. Run tests with `pytest`:

   ```bash
   poetry run pytest
   ```

3. Debug test failures:
   Run `pytest` with verbose output:

   ```bash
   poetry run pytest -v
   ```

4. Generate a test coverage report:
   Install `pytest-cov`:

   ```bash
   poetry add pytest-cov --group dev
   ```

   Run tests with coverage:

   ```bash
   poetry run pytest --cov=wtv
   ```

### Testing Documentation Locally

To test the documentation locally, follow these steps:

1. **Install MkDocs**:
   Ensure MkDocs and its dependencies are installed. run:

   ```bash
   poetry install --with docs
   ```

2. **Serve the Documentation**:
   Use the following command to serve the documentation locally:

   ```bash
   poetry run mkdocs serve
   ```

3. **Access the Documentation**:
   Open your browser and navigate to `http://127.0.0.1:8000` to view the documentation.

### Running GitHub Actions Locally with `act`

To run all GitHub Actions workflows locally using `act`, follow these steps:

1. **Install act**:
   Download and install `act` from its [GitHub repository](https://github.com/nektos/act).

2. **Set Up Secrets**:
   Create a `.secrets` file in the root of your repository and define the required secrets. For example:

   ```
   PYPI_API_TOKEN=your-real-or-mock-token-for-testing
   ```

3. **Run All Workflows**:
   Use the following command to run all workflows:

   ```bash
   act
   ```

4. **Run a Specific Workflow**:
   To run a specific workflow, use the `-W` flag followed by the path to the workflow file:

   ```bash
   act -W .github/workflows/package.yaml
   act -W .github/workflows/publish.yml
   ```

5. **Specify an Event**:
   If you want to simulate a specific event (e.g., `push`, `pull_request`, or `release`), use the `-e` flag:

   ```bash
   act -e push
   ```

6. **Use a Specific Runner**:
   By default, `act` uses a lightweight Docker image. To use a full-featured image (e.g., `ubuntu-latest`), specify it:
   ```bash
   act -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-latest
   ```

#### Example `act` Commands

- Run all workflows:

  ```bash
  act
  ```

- Run the `package.yaml` workflow:

  ```bash
  act -W .github/workflows/package.yaml
  ```

- Run the `publish.yml` workflow:

  ```bash
  act -W .github/workflows/publish.yml
  ```

- Simulate a `release` event:
  ```bash
  act -e release
  ```

### Documentation Deployment

This project uses GitHub Actions to auto-generate and deploy documentation to GitHub Pages. To enable this workflow, ensure the following secret is configured in your repository:

- **`GITHUB_TOKEN`**: This is automatically provided by GitHub for workflows. No additional setup is required unless you are using a custom token.

### Known Issues

- It seems to generate slightly different results based on the OS version and/or Python version.

### Acknowledgements

This project is based on the original work by Honglun Yuan, Yiding Jiangfang, Zhenhua Liu, Rong Rong Su, Qiao Li, Chuanying Fang, Sishu Huang, Xianqing Liu, Alisdair Robert Fernie, and Jie Luo, as published in [WTV_2.0](https://github.com/yuanhonglun/WTV_2.0) and [their associated publication](https://doi.org/10.1016/j.molp.2024.04.012).
