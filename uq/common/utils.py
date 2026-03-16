from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory."""
    # Try to find repo root by looking for pyproject.toml
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Max 10 levels up
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to cwd
    return Path.cwd()