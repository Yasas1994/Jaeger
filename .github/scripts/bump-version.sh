#!/usr/bin/env bash
#
# Bump the version across all project files.
#
# Versioning scheme: <major>.<year>.<update>
#   e.g. 1.26.1  -> major=1, year=26 (2026), update=1
#
# Usage:
#   ./bump-version.sh [major] [year] [update]
#
# If arguments are omitted, the script auto-increments:
#   - If the current year matches <year>, increment <update> by 1
#   - If the year changed, reset <update> to 1
#
# Examples:
#   ./bump-version.sh           # auto-bump (e.g. 1.26.1 -> 1.26.2)
#   ./bump-version.sh 1 26 5    # explicit (sets 1.26.5)
#   ./bump-version.sh 2 27 1    # new major, new year

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Parse current version from pyproject.toml
# ---------------------------------------------------------------------------
CURRENT_VERSION=$(grep -m1 '^version\s*=' "${REPO_ROOT}/pyproject.toml" | sed 's/.*"\([^"]*\)".*/\1/')
if [[ -z "${CURRENT_VERSION}" ]]; then
    echo "Error: could not parse version from pyproject.toml" >&2
    exit 1
fi

# Strip any PEP 440 local segment (+...) or pre-release suffix for parsing
PARSE_VERSION="${CURRENT_VERSION%%+*}"
PARSE_VERSION="${PARSE_VERSION%%[abrc]*}"

IFS='.' read -r CUR_MAJOR CUR_YEAR CUR_UPDATE <<< "${PARSE_VERSION}"

echo "Current version: ${CURRENT_VERSION}"
echo "  major=${CUR_MAJOR}, year=${CUR_YEAR}, update=${CUR_UPDATE}"

# ---------------------------------------------------------------------------
# Determine new version
# ---------------------------------------------------------------------------
if [[ $# -eq 3 ]]; then
    NEW_MAJOR="$1"
    NEW_YEAR="$2"
    NEW_UPDATE="$3"
elif [[ $# -eq 0 ]]; then
    THIS_YEAR_SHORT="$(date +%y)"
    if [[ "${CUR_YEAR}" == "${THIS_YEAR_SHORT}" ]]; then
        NEW_MAJOR="${CUR_MAJOR}"
        NEW_YEAR="${CUR_YEAR}"
        NEW_UPDATE=$((CUR_UPDATE + 1))
    else
        NEW_MAJOR="${CUR_MAJOR}"
        NEW_YEAR="${THIS_YEAR_SHORT}"
        NEW_UPDATE=1
    fi
else
    echo "Usage: $0 [major year update]" >&2
    echo "  e.g. $0 1 26 3   -> sets version to 1.26.3" >&2
    echo "  e.g. $0          -> auto-increment" >&2
    exit 1
fi

NEW_VERSION="${NEW_MAJOR}.${NEW_YEAR}.${NEW_UPDATE}"
NEW_TAG="v${NEW_VERSION}"

echo ""
echo "New version: ${NEW_VERSION}"
echo "  major=${NEW_MAJOR}, year=${NEW_YEAR}, update=${NEW_UPDATE}"

# ---------------------------------------------------------------------------
# Helper: sed that works on both GNU and BSD sed
# ---------------------------------------------------------------------------
sed_i() {
    local expr="$1"
    local file="$2"
    if sed --version >/dev/null 2>&1; then
        # GNU sed
        sed -i "$expr" "$file"
    else
        # BSD sed (macOS)
        sed -i '' "$expr" "$file"
    fi
}

# ---------------------------------------------------------------------------
# Update files
# ---------------------------------------------------------------------------

echo ""
echo "Updating files..."

# 1. pyproject.toml
sed_i "s/^version = \"[^\"]*\"/version = \"${NEW_VERSION}\"/" "${REPO_ROOT}/pyproject.toml"
echo "  ✓ pyproject.toml"

# 2. .cz.toml
if [[ -f "${REPO_ROOT}/.cz.toml" ]]; then
    sed_i "s/^version = \"[^\"]*\"/version = \"${NEW_VERSION}\"/" "${REPO_ROOT}/.cz.toml"
    echo "  ✓ .cz.toml"
fi

# 3. recipes/jaeger-bio/meta.yaml — version only (deps updated separately)
META_YAML="${REPO_ROOT}/recipes/jaeger-bio/meta.yaml"
if [[ -f "${META_YAML}" ]]; then
    sed_i "s/{% set version = \"[^\"]*\" %}/{% set version = \"${NEW_VERSION}\" %}/" "${META_YAML}"
    echo "  ✓ recipes/jaeger-bio/meta.yaml (version)"
fi

# 3b. Sync meta.yaml run dependencies from pyproject.toml
UPDATE_DEPS_SCRIPT="${REPO_ROOT}/.github/scripts/update-meta-deps.py"
if [[ -f "${UPDATE_DEPS_SCRIPT}" ]] && [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
    (cd "${REPO_ROOT}" && python3 "${UPDATE_DEPS_SCRIPT}")
fi

# 4. Singularity definitions — update pinned pip install version
for def_file in "${REPO_ROOT}"/singularity/*.def; do
    if [[ -f "$def_file" ]]; then
        # Match "jaeger-bio==X.Y.Z" or "jaeger-bio==X.Y.ZbN"
        if grep -q 'jaeger-bio==[0-9]\+\.[0-9]\+\.[0-9]\+.*"' "$def_file" 2>/dev/null; then
            sed_i 's/jaeger-bio==[0-9]\+\.[0-9]\+\.[0-9]\+[a-zA-Z0-9]*/jaeger-bio=='"${NEW_VERSION}"'/' "$def_file"
            echo "  ✓ $(basename "$def_file") (pinned version)"
        fi
    fi
done

# 5. README.md
README="${REPO_ROOT}/README.md"
if [[ -f "$README" ]]; then
    # 5a. Header line "## Jaeger X.Y.Z" (if present)
    if grep -q '^## Jaeger [0-9]\+\.[0-9]\+\.[0-9]\+' "$README" 2>/dev/null; then
        sed_i 's/^## Jaeger [0-9]\+\.[0-9]\+\.[0-9]\+[a-zA-Z0-9]*/## Jaeger '"${NEW_VERSION}"'/' "$README"
        echo "  ✓ README.md (header)"
    fi

    # 5b. Pinned install command: jaeger-bio==X.Y.Z
    if grep -q 'jaeger-bio==[0-9]\+\.[0-9]\+\.[0-9]\+' "$README" 2>/dev/null; then
        sed_i 's/jaeger-bio==[0-9]\+\.[0-9]\+\.[0-9]\+[a-zA-Z0-9]*/jaeger-bio=='"${NEW_VERSION}"'/g' "$README"
        echo "  ✓ README.md (pinned install version)"
    fi
fi

# 6. AGENTS.md — update the "Current version" line
AGENTS="${REPO_ROOT}/AGENTS.md"
if [[ -f "$AGENTS" ]]; then
    if grep -q 'Current version' "$AGENTS" 2>/dev/null; then
        sed_i 's/Current version.*`[0-9]\+\.[0-9]\+\.[0-9]\+[a-zA-Z0-9]*`/Current version: `'"${NEW_VERSION}"'`/' "$AGENTS"
        echo "  ✓ AGENTS.md"
    fi
fi

# 7. docs/_source/usage.md and docs/_source/_static/usage.md
for usage_file in "${REPO_ROOT}/docs/_source/usage.md" "${REPO_ROOT}/docs/_source/_static/usage.md"; do
    if [[ -f "$usage_file" ]]; then
        # Update header line
        if grep -q '^## Jaeger [0-9]\+\.[0-9]\+\.[0-9]\+' "$usage_file" 2>/dev/null; then
            sed_i 's/^## Jaeger [0-9]\+\.[0-9]\+\.[0-9]\+[a-zA-Z0-9]*/## Jaeger '"${NEW_VERSION}"'/' "$usage_file"
            echo "  ✓ $(basename "$usage_file") (header)"
        fi
        # Update singularity filename references like jaeger_1.1.30.sif
        if grep -q 'jaeger_[0-9]\+\.[0-9]\+\.[0-9]\+\.sif' "$usage_file" 2>/dev/null; then
            sed_i 's/jaeger_[0-9]\+\.[0-9]\+\.[0-9]\+[a-zA-Z0-9]*\.sif/jaeger_'"${NEW_VERSION}"'.sif/g' "$usage_file"
            echo "  ✓ $(basename "$usage_file") (singularity filename)"
        fi
    fi
done

# 8. docs/_source/changelog.md — update top header if it matches current version
CHANGELOG_COPY="${REPO_ROOT}/docs/_source/changelog.md"
if [[ -f "$CHANGELOG_COPY" ]]; then
    # Only update the topmost version header, not historical ones
    # We leave historical changelog entries untouched
    echo "  ✓ docs/_source/changelog.md (historical entries preserved)"
fi

# 9. CHANGELOG.md — commitizen will handle this, but we note it
CHANGELOG="${REPO_ROOT}/CHANGELOG.md"
if [[ -f "$CHANGELOG" ]]; then
    echo "  ✓ CHANGELOG.md (run 'cz bump' to regenerate)"
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo ""
echo "Verification:"
grep -m1 '^version\s*=' "${REPO_ROOT}/pyproject.toml" 2>/dev/null || true
if [[ -f "${REPO_ROOT}/.cz.toml" ]]; then
    grep -m1 '^version\s*=' "${REPO_ROOT}/.cz.toml" 2>/dev/null || true
fi
if [[ -f "${META_YAML}" ]]; then
    grep -m1 'set version' "${META_YAML}" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Git commit reminder
# ---------------------------------------------------------------------------
echo ""
echo "Done. Next steps:"
echo "  1. Review the changes: git diff"
echo "  2. Stage and commit:  git add -A && git commit -m \"chore(release): bump version to ${NEW_VERSION}\""
echo "  3. Tag the release:   git tag ${NEW_TAG}"
echo "  4. Push:              git push origin main --tags"
echo ""
echo "Or use commitizen:     cz bump --${NEW_VERSION}"
