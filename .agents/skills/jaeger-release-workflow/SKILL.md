---
name: jaeger-release-workflow
description: Guide Kimi through the Jaeger (jaeger-bio) release process. Use when the user wants to bump the version, cut a new release, publish to PyPI, sync remotes, update the Bioconda recipe, build/test the recipe locally, or open the Bioconda PR for the Jaeger project.
---

# Jaeger Release Workflow

Release `jaeger-bio` end-to-end: version bump → GitHub release → PyPI → Bioconda recipe → Bioconda PR.

Assumptions:
- Local clone is at `/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/Jaeger`.
- Fork of `bioconda-recipes` is at `/home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/bioconda-recipes`.
- Remotes `Yasas1994` and `MGXlab` point to the Jaeger forks.
- `gh` CLI is authenticated as `Yasas1994`.

## 1. Bump version

Install commitizen in `jaeger_dev` if missing, then run the bump:

```bash
source /home/yasas-wijesekara/.miniforge3/etc/profile.d/conda.sh
conda activate jaeger_dev
pip install commitizen
cd /home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/Jaeger
cz bump <VERSION>
```

This updates `pyproject.toml`, `.cz.toml`, `CHANGELOG.md`, and creates `v<VERSION>`.

If `pyproject.toml` is not updated, add to `.cz.toml`:

```toml
version_files = [
    "pyproject.toml:version",
]
```

## 2. Push main and tag to both forks

```bash
git push Yasas1994 main --tags
git push MGXlab main --tags
```

## 3. Create GitHub release

```bash
gh release create v<VERSION> \
  --repo Yasas1994/Jaeger \
  --title "Jaeger <VERSION>" \
  --generate-notes
```

This triggers `.github/workflows/publish-to-pypi.yml`.

## 4. Update Bioconda recipe

The `.github/workflows/bioconda-update.yml` workflow should create a branch `bump-jaeger-bio-<VERSION>` on `Yasas1994/bioconda-recipes`. If it fails because the branch already exists, ensure the workflow force-pushes the ephemeral branch.

Manually verify the recipe dependencies match `pyproject.toml`:

- Add missing direct runtime imports (e.g., `h5py`, `joblib`, `numpy`, `requests`, `rich`, `scipy`, `seaborn`).
- Keep `tensorflow` unpinned (`- tensorflow`) because conda-forge lags behind the pip pin.
- Remove unused dependencies such as `progressbar2`.

Edit both:
- `recipes/jaeger-bio/meta.yaml` in the Jaeger repo
- `recipes/jaeger-bio/meta.yaml` in the `bump-jaeger-bio-<VERSION>` branch of `bioconda-recipes`

## 5. Build and test the Bioconda recipe locally

```bash
source /home/yasas-wijesekara/.miniforge3/etc/profile.d/conda.sh
conda activate bioconda
export PATH="/home/yasas-wijesekara/.pixi/envs/bioconda-utils/bin:$PATH"
cd /home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/bioconda-recipes
bioconda-utils build --n-workers 20 --mulled-test --packages jaeger-bio recipes/jaeger-bio/meta.yaml
```

Expected result: `BUILD SUMMARY: successfully built 1 of 1 recipes` and `TEST SUCCESS recipes/jaeger-bio`.

## 6. Push the Bioconda branch and open the PR

```bash
cd /home/yasas-wijesekara/ssd/Projects/Jaeger_revisions/bioconda-recipes
git push origin bump-jaeger-bio-<VERSION> --force-with-lease

gh pr create \
  --repo bioconda/bioconda-recipes \
  --head Yasas1994:bump-jaeger-bio-<VERSION> \
  --base master \
  --title "Bump jaeger-bio to <VERSION>" \
  --body "Update jaeger-bio to version <VERSION>."
```

## 7. Install the local build for manual testing (optional)

```bash
conda create -n jaeger_test \
  -c local -c conda-forge -c bioconda \
  jaeger-bio=<VERSION>
conda activate jaeger_test
jaeger --version
jaeger health
```

## Common fixes

- **PyPI upload fails with "File already exists"**: `cz bump` did not update `pyproject.toml`. Add `version_files`, re-bump, move the tag, force-push, and recreate the release.
- **Bioconda build fails with `ModuleNotFoundError: tensorflow`**: add `- tensorflow` to the recipe run deps.
- **Bioconda branch push rejected**: use `git push --force-with-lease origin bump-jaeger-bio-<VERSION>`.
- **Lint/format errors**: run `ruff check src/jaeger` and `ruff format src/jaeger` before committing.
