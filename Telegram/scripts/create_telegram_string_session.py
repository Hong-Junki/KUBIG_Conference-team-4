import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main():
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local")

    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        raise SystemExit("Set TELEGRAM_API_ID and TELEGRAM_API_HASH first.")

    try:
        from telethon.sync import TelegramClient
        from telethon.sessions import StringSession
    except ImportError:
        raise SystemExit("Missing dependency: telethon. Run: pip install -r requirements.txt")

    with TelegramClient(StringSession(), int(api_id), api_hash) as client:
        print("\nSave this value as GitHub Secret TELEGRAM_STRING_SESSION:\n")
        print(client.session.save())
        print("\nDo not commit or share this value.")


if __name__ == "__main__":
    main()
