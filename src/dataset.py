from pathlib import Path

def load_lyrics_text(path: str = "data/lyrics.txt") -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {p.resolve()}. Put your corpus in data/lyrics.txt")
    return p.read_text(encoding="utf-8", errors="ignore")
