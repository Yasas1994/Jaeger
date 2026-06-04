#!/usr/bin/env python3
"""Update bioconda meta.yaml run dependencies from pyproject.toml."""

import re
import sys
from pathlib import Path

# pip -> conda package name mapping
PIP_TO_CONDA = {
    "matplotlib": "matplotlib-base",
    "parasail": "parasail-python",
}


def parse_pyproject_deps(pyproject_path: Path) -> list[tuple[str, str]]:
    """Extract dependencies from pyproject.toml."""
    content = pyproject_path.read_text()

    match = re.search(r"dependencies = \[(.*?)\]", content, re.DOTALL)
    if not match:
        print("Error: Could not find dependencies block", file=sys.stderr)
        sys.exit(1)

    deps = []
    for line in match.group(1).split("\n"):
        line = line.strip().strip(",").strip('"')
        if not line:
            continue
        # Handle optional extras like "tensorflow[and-cuda] >=2.18, <2.19"
        m = re.match(r'^([a-zA-Z0-9_-]+)(?:\[[^\]]+\])?\s*([<>=!~].*)?$', line)
        if m:
            deps.append((m.group(1), (m.group(2) or "").strip()))

    return deps


def pip_to_conda_dep(name: str, spec: str) -> str:
    conda_name = PIP_TO_CONDA.get(name, name)
    # Conda requires no spaces around version operators
    # e.g., pip: "keras >= 3.12.0" -> conda: "keras >=3.12.0"
    spec = spec.replace(", ", ",")
    spec = re.sub(r'\s*([<>=!~]+)\s*', r'\1', spec)
    return f"    - {conda_name} {spec}" if spec else f"    - {conda_name}"


def update_meta_yaml(meta_path: Path, deps: list[tuple[str, str]]) -> None:
    content = meta_path.read_text()

    run_lines = [
        "    - python >=3.11,<3.13",
        "    - pip",
        "    # Core dependencies synced from pyproject.toml",
    ]
    for name, spec in deps:
        run_lines.append(pip_to_conda_dep(name, spec))
    new_run = "\n".join(run_lines)

    # Replace run: section, preserving the following test: section
    # The test: section may be at same indent (2 spaces) or no indent
    pattern = r"(  run:\n)(.*?)(\n\ntest:|\n  test:)"

    def replacer(m):
        return m.group(1) + new_run + m.group(3)

    new_content = re.sub(pattern, replacer, content, flags=re.DOTALL)

    if new_content == content:
        print("Warning: Could not find run: section to update", file=sys.stderr)
        sys.exit(1)

    meta_path.write_text(new_content)
    print(f"  ✓ {meta_path} (dependencies synced from pyproject.toml)")


def main():
    repo_root = Path.cwd()
    pyproject = repo_root / "pyproject.toml"
    meta_yaml = repo_root / "recipes" / "jaeger-bio" / "meta.yaml"

    if not pyproject.exists():
        print(f"Error: {pyproject} not found", file=sys.stderr)
        sys.exit(1)

    if not meta_yaml.exists():
        print(f"Error: {meta_yaml} not found", file=sys.stderr)
        sys.exit(1)

    deps = parse_pyproject_deps(pyproject)
    update_meta_yaml(meta_yaml, deps)


if __name__ == "__main__":
    main()
