from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "outputs"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json"}
FORBIDDEN = (
    "ming" + "chen",
    "pretrain" + "_weight/model.pt",
    "E:\\vggt" + "_research",
    "C:\\Users" + "\\",
)


def test_repo_does_not_embed_old_private_paths() -> None:
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for token in FORBIDDEN:
            if token.lower() in text:
                hits.append(f"{path.relative_to(ROOT)}: {token}")
    assert hits == []
