# Releasing

How the version is stored, how to bump it, and how a release reaches PyPI.

## Where the version lives

The version is declared in exactly one place: `[project].version` in
`pyproject.toml` (PEP 621). Everything else is derived from it.

| Place                                   | How it gets the version                                            |
| --------------------------------------- | ------------------------------------------------------------------ |
| `pyproject.toml`, `[project].version`   | declared by hand - the only source of truth                        |
| `wtv.__version__`                       | read from the installed distribution metadata (`importlib.metadata`) |
| `wtv-cli --version`                     | the same metadata, through `wtv.__version__`                       |
| `uv.lock`, the `name = "wtv"` package   | written by uv when the lock is refreshed                           |
| git tag `v<version>`                    | created when the release is cut                                    |

Never edit the version in `uv.lock` by hand - refresh the lock with `uv lock`.
`tests/test_version.py` fails if the declared version, the installed metadata and
the lock file disagree, so a mismatch cannot go unnoticed.

## Version policy

The project adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html),
as recorded at the top of the changelog.

| Change                                                                     | Bump  | Changelog section |
| -------------------------------------------------------------------------- | ----- | ----------------- |
| Bug fix with no change in behaviour                                         | patch | Fixed             |
| New backwards-compatible functionality, a new CLI option or API parameter   | minor | Added             |
| A change to what the same input produces, e.g. a rescaled default parameter | minor | Changed           |
| Removed or renamed option or parameter, dropped Python version, new output layout | major | Changed / Removed |

Because a changed default can change a user's results, it is never a patch bump.
Whatever the bump, the change and its reason belong in the changelog.

## Bumping the version

Bumping is done with [bump-my-version](https://callowayproject.github.io/bump-my-version/),
configured in `[tool.bumpversion]` in `pyproject.toml`. It reads the current
version from the PEP 621 metadata, so there is no second copy of the version to
keep in sync, and it rewrites `pyproject.toml` and `uv.lock` together. It does not
commit or tag - the release branch goes through a pull request like any other
change.

1.  Start from a clean, up-to-date `main`. The tool refuses a tree with modified
    tracked files, which is why the setup for a release is committed before the
    bump itself. Untracked files are not refused, so do not expect it to catch
    stray data files for you.

    ```bash
    git checkout main && git pull
    git checkout -b rel/<version>
    ```

2.  Preview what would change - `<part>` is `major`, `minor` or `patch`:

    ```bash
    uv run bump-my-version bump <part> --dry-run --verbose
    ```

    Two targets should be listed: `pyproject.toml:project.version`, printed as a
    PEP 621 pointer rather than a bare filename, and `uv.lock`. Anything else in
    that list is a mistake in `[tool.bumpversion]`.

3.  Apply the bump:

    ```bash
    uv run bump-my-version bump <part>
    ```

4.  Refresh the environment, so the installed metadata and the lock agree:

    ```bash
    uv sync --all-extras --dev
    ```

    Without this step the editable install still reports the previous version and
    `tests/test_version.py` fails. That failure is the intended signal, not a
    problem with the test.

    Pass the extras. `dev` and `docs` are optional-dependency extras rather than
    uv dependency groups, so a bare `uv sync` removes them - it uninstalls pytest,
    ruff, black, isort and bump-my-version, and the verification below then fails
    for the wrong reason.

5.  Verify:

    ```bash
    uv lock --locked          # the lock agrees with pyproject.toml
    uv run pytest             # the whole suite, including the version checks
    uv run ruff check .
    uv build                  # dist/wtv-<version>-*.whl and .tar.gz carry the new version
    uv run wtv-cli --version
    ```

6.  Move the `[Unreleased]` changelog block under a `## [<version>] - YYYY-MM-DD`
    heading and open a fresh, empty `[Unreleased]`. Stamp the date of preparation
    and correct it if the release slips.

7.  Commit the two rewritten files plus the changelog, push the branch and open a
    pull request against `main`. CI runs the tests and the build on three
    operating systems.

## Cutting the release

After the release pull request is merged:

```bash
git checkout main && git pull
git tag -a v<version> -m "Release <version>"
git push origin v<version>
```

`-a` makes an annotated tag. The tags up to v0.2.1 are lightweight, so this
deliberately changes the convention: an annotated tag records who tagged it and
when, and `git describe` ignores lightweight tags unless given `--tags`.

Then publish a GitHub release for that tag. `.github/workflows/publish.yml` runs
on the `published` event, builds with `uv build` and uploads with
`uv publish --token $PYPI_API_TOKEN`. The workflow can also be started by hand
(`Run workflow`) if the release already exists and publishing failed.

## Checklist

- [ ] The bump level matches the policy above, not merely the size of the diff
- [ ] The dry run lists only `pyproject.toml:project.version` and `uv.lock`
- [ ] The environment was refreshed with `uv sync --all-extras --dev`, so the
      verification ran with the tools installed
- [ ] Every user-visible change since the previous tag is in the changelog section
      for this release, including its reason and any limitation
- [ ] `uv lock --locked`, `uv run pytest`, `uv run ruff check .` and `uv build` pass
- [ ] The release pull request is merged into `main` before the tag is pushed
- [ ] The GitHub release notes match the changelog section
- [ ] After publishing, the new version installs from PyPI:
      `pip install wtv==<version> && wtv-cli --version`

## Limitations

- Publishing requires a published GitHub release; pushing a tag alone publishes
  nothing.
- The bioconda recipe advertised by the README badge lives outside this
  repository and is not covered by this workflow.
- Changelog entries are written by hand; nothing generates them from merged pull
  requests.
- The `uv.lock` entry under `[tool.bumpversion]` matches the shape of the lock
  file uv writes today. If that shape changes, the bump stops with an error
  (`ignore_missing_version = false`) rather than half-bumping the release.
- `dev` is a published extra, not a uv dependency group, so `bump-my-version` and
  the other release tools appear in `Requires-Dist ... extra == "dev"` and
  `pip install wtv[dev]` pulls them. Moving the whole `dev` extra to a
  `[dependency-groups]` section would keep them out of the published metadata and
  make a bare `uv sync` install them; that changes every `uv sync --all-extras`
  call in the README, the docs and CI, so it is a change for its own branch.
