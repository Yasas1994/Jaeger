# Releasing Jaeger

This guide is for maintainers who need to bump the version, build a new release, and publish it to PyPI and Bioconda.

---

## Versioning scheme

Jaeger uses a calendar-based versioning format:

```
<major>.<year>.<update>
```

| Component | Meaning | Example |
|---|---|---|
| `major` | Major product version | `1` |
| `year` | Last two digits of release year | `26` = 2026 |
| `update` | Incremental update within that year | `1`, `2`, `3` … |

Examples:
- `1.26.1` — first update of 2026
- `1.26.5` — fifth update of 2026
- `2.27.1` — first update of 2027, new major version

---

## Prerequisites

- Commitizen (`cz`) installed: `pip install commitizen`
- PDM installed: `pip install pdm`
- Write access to the canonical repository (`Yasas1994/Jaeger`)
- PyPI Trusted Publishing (OIDC) configured for the repository (see [OIDC setup](#openid-connect-oidc-setup) below)
- Bioconda bot token (`BIOCONDA_BOT_TOKEN`) configured as a repository secret

---

## Step 1: Bump the version

### Option A — Auto-bump (recommended)

Run the bump script from the repo root. It auto-detects the current version and increments the `update` counter (or resets to `1` if the year changed):

```bash
.github/scripts/bump-version.sh
```

### Option B — Explicit version

Set a specific version directly:

```bash
.github/scripts/bump-version.sh 1 26 5   # sets 1.26.5
.github/scripts/bump-version.sh 2 27 1   # sets 2.27.1
```

### What the script updates

The script edits version strings in:

| File | Field |
|---|---|
| `pyproject.toml` | `version = "…"` |
| `.cz.toml` | `version = "…"` |
| `recipes/jaeger-bio/meta.yaml` | `{% set version = "…" %}` |
| `singularity/jaeger_singularity.def` | `jaeger-bio==…` |
| `README.md` | `## Jaeger X.Y.Z` header (if present) + `jaeger-bio==…` |
| `AGENTS.md` | ``Current version: `X.Y.Z` `` |
| `docs/_source/usage.md` | Header + `jaeger_X.Y.Z.sif` |
| `docs/_source/_static/usage.md` | Header + `jaeger_X.Y.Z.sif` |

Review the changes:

```bash
git diff
```

---

## Step 2: Commit and tag

Stage the version bumps and commit:

```bash
git add -A
git commit -m "chore(release): bump version to $(grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')"
```

Create an annotated tag:

```bash
git tag -a v$(grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)".*/\1/') -m "Release version X.Y.Z"
```

Or use Commitizen to handle changelog + tag together:

```bash
cz bump
```

This updates `CHANGELOG.md`, bumps `.cz.toml`, and creates the Git tag automatically.

---

## Step 3: Push to trigger CI

Push the commit and tag to the **canonical repository** (`Yasas1994/Jaeger`):

```bash
git push origin main
git push origin --tags
```

> **Important:** The publish and release workflows are gated with `github.repository == 'Yasas1994/Jaeger'`. Pushing to forks (e.g. `MGXlab/Jaeger`) will build artifacts for validation but will **not** publish to PyPI or Bioconda.

---

## Step 4: What happens automatically

Once the tag is pushed, GitHub Actions runs the following workflows in sequence:

### 1. `publish-to-pypi.yaml`
- **Builds** sdist + wheel via `pdm build`
- **Publishes to PyPI** (production) — only from `Yasas1994/Jaeger`
- Uses **Trusted Publishing (OIDC)** — no API tokens needed

### 2. `release.yaml`
- Creates a **GitHub Release** from the tag
- Pulls release notes from the matching section in `CHANGELOG.md`
- Auto-detects pre-releases (`a`, `b`, `rc` in tag name)

### 3. `bioconda-update.yaml`
- Triggered by the GitHub Release being **published**
- Downloads the PyPI tarball and computes SHA256
- Checks out `bioconda/bioconda-recipes`
- Updates `recipes/jaeger-bio/meta.yaml` (version + SHA256)
- Pushes a branch and opens a PR to Bioconda

---

## Step 5: Verify the release

### PyPI

Check that the new version appears on:

```
https://pypi.org/project/jaeger-bio/
```

Install and test:

```bash
pip install --upgrade jaeger-bio
jaeger --version
```

### Bioconda

Monitor the automated PR at:

```
https://github.com/bioconda/bioconda-recipes/pulls?q=is%3Apr+jaeger-bio
```

Once merged, the new version will be available via:

```bash
conda install -c bioconda jaeger-bio
```

### GitHub Release

Check the release page:

```
https://github.com/Yasas1994/Jaeger/releases
```

---

## OpenID Connect (OIDC) setup

PyPI Trusted Publishing uses OIDC so GitHub Actions can authenticate without storing long-lived API tokens.

### 1. Create a PyPI project (or claim an existing one)

If the project already exists on PyPI, log in as an owner or maintainer:

```
https://pypi.org/manage/project/jaeger-bio/settings/
```

### 2. Add a Trusted Publisher

In **Publishing**, click **Add a new pending publisher** and fill:

| Field | Value |
|---|---|
| **Publisher type** | GitHub Actions |
| **Owner** | `Yasas1994` |
| **Repository name** | `Jaeger` |
| **Workflow name** | `publish-to-pypi.yaml` |
| **Environment name** | `pypi` *(optional but recommended)* |

Click **Add**. The publisher is now pending — it becomes active after the first successful publish.

### 3. Add TestPyPI publisher (optional but recommended)

Repeat the same on TestPyPI:

```
https://test.pypi.org/manage/project/jaeger-bio/settings/
```

Use environment name `testpypi`.

### 4. How it works in the workflow

The workflow requests a short-lived OIDC token from GitHub:

```yaml
permissions:
  id-token: write   # Required for OIDC

environment:
  name: pypi        # Must match the PyPI environment name
```

`pypa/gh-action-pypi-publish` exchanges this token for a PyPI upload token automatically. No `PYPI_API_TOKEN` secret is needed.

### 5. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `trusted publishing exchange failed` | Publisher not configured on PyPI | Add the pending publisher in PyPI settings |
| `invalid-publisher: invalid token` | Wrong owner/repo/workflow name | Double-check the exact values in PyPI settings |
| `environment … not found` | `environment:` name mismatch | Ensure the workflow `environment.name` matches PyPI |
| `permission denied` | Running on a fork | Only the canonical repo (`Yasas1994/Jaeger`) is authorized |

---

## Manual fallback

If any automated step fails, you can trigger it manually:

### Manual PyPI publish

```bash
pdm build
# Upload to TestPyPI first
pdm publish --repository testpypi
# Then production
pdm publish
```

### Manual Bioconda update

Go to **Actions → Bioconda Recipe Update → Run workflow** and enter the version:

```
version: 1.26.5
```

This requires the `BIOCONDA_BOT_TOKEN` secret to be configured.

### Manual GitHub Release

Go to **Releases → Draft a new release**, select the tag, and publish.

---

## Repository guard

The following jobs are restricted to the canonical repository:

| Workflow | Job | Guard |
|---|---|---|
| `publish-to-pypi.yaml` | `publish-testpypi` | `github.repository == 'Yasas1994/Jaeger'` |
| `publish-to-pypi.yaml` | `publish-pypi` | `github.repository == 'Yasas1994/Jaeger'` |
| `release.yaml` | `release` | `github.repository == 'Yasas1994/Jaeger'` |
| `bioconda-update.yaml` | `update-bioconda` | `github.repository == 'Yasas1994/Jaeger'` |

If you maintain a fork and want it to become the canonical repo, change the guard string in all four workflow files.
