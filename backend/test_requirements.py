from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def packages(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def test_vercel_requirements_match_backend():
    assert packages(ROOT / "requirements.txt") == packages(ROOT / "backend" / "requirements.txt")
