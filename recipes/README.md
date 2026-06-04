# Bioconda Recipe Update Workflow

Guide for contributing fixes to a biocondabot-created branch in `bioconda/bioconda-recipes`.

> **Note:** Branch name for jaeger-bio is `bump/jaeger_bio` (underscore, not hyphen).

---

## 1. Fork & Clone

Go to https://github.com/bioconda/bioconda-recipes and click **Fork** (if not already done).

```bash
git clone https://github.com/Yasas1994/bioconda-recipes.git
cd bioconda-recipes
```

---

## 2. Set Up Remotes

```bash
git remote add upstream https://github.com/bioconda/bioconda-recipes.git
git remote -v
# origin    https://github.com/Yasas1994/bioconda-recipes.git
# upstream  https://github.com/bioconda/bioconda-recipes.git
```

---

## 3. Pull in the Biocondabot Branch

```bash
git fetch upstream
git checkout -b bump/jaeger_bio upstream/bump/jaeger_bio
```

---

## 4. Make Your Fixes

Edit `recipes/jaeger-bio/meta.yaml` — add `pdm-backend` and pin `setuptools` to the host requirements:

```yaml
requirements:
  host:
    - python
    - pip
    - pdm-backend        # required build backend
    - setuptools <81     # pin to avoid build breakage
  run:
    - python
```

> **Why?** Bioconda's conda-build sandbox blocks PyPI, so any build backend (like `pdm-backend`)
> must be explicitly listed in `host` rather than relying on pip to fetch it at build time.

---

## 5. Commit

```bash
git add recipes/jaeger-bio/meta.yaml
git commit -m "fix: add pdm-backend to host requirements"
```

---

## 6. Push to Your Fork

```bash
git push origin bump/jaeger_bio
```

---

## 7. Open a PR Targeting the Bot's Branch

```bash
gh pr create \
  --repo bioconda/bioconda-recipes \
  --title "Fix jaeger-bio build: add pdm-backend" \
  --body "Adds missing pdm-backend host dependency to fix BackendUnavailable error during conda build." \
  --head "Yasas1994:bump/jaeger_bio" \
  --base bump/jaeger_bio
```

Or manually on GitHub:
- Go to https://github.com/bioconda/bioconda-recipes/compare
- Set **base repository** → `bioconda/bioconda-recipes`, **base** → `bump/jaeger_bio`
- Set **head repository** → `Yasas1994/bioconda-recipes`, **compare** → `bump/jaeger_bio`
- Click **Create pull request**

---

## Keeping Your Branch Up to Date

If further changes are pushed to the upstream bot branch:

```bash
git fetch upstream
git rebase upstream/bump/jaeger_bio
git push origin bump/jaeger_bio --force-with-lease
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `BackendUnavailable: Cannot import 'pdm.backend'` | `pdm-backend` not in conda host deps | Add `pdm-backend` to `host` in `meta.yaml` |
| `remote: Permission denied (403)` | Pushing to upstream instead of fork | Push to `origin` (your fork), not `upstream` |
| `setuptools` version conflict | setuptools ≥81 breaks build | Add `setuptools <81` to `host` in `meta.yaml` |
