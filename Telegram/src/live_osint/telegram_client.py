from __future__ import annotations

import os
from pathlib import Path


def load_local_env(config_path: Path) -> None:
    root = config_path.resolve().parent.parent
    for env_path in [root / ".env.local", root / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_channel_config(config_path: Path) -> list[dict]:
    import json

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return [c for c in payload.get("channels", []) if c.get("enabled", True)]


async def collect_with_telethon(config_path: Path, limit_per_channel: int) -> list[dict]:
    try:
        from telethon import TelegramClient
        from telethon.sessions import StringSession
    except ImportError as exc:
        raise RuntimeError("Telethon is not installed. Run: pip install telethon") from exc

    load_local_env(config_path)

    root = config_path.resolve().parent.parent
    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    string_session = os.environ.get("TELEGRAM_STRING_SESSION") or os.environ.get("TELEGRAM_SESSION_STRING")
    session = os.environ.get("TELEGRAM_SESSION", "kubig_live_osint")
    session_path = Path(session)
    if not session_path.is_absolute():
        session_path = root / session_path
    if not api_id or not api_hash:
        raise RuntimeError("Set TELEGRAM_API_ID and TELEGRAM_API_HASH before collecting Telegram messages.")

    channels = load_channel_config(config_path)
    if not channels:
        raise RuntimeError(f"No enabled channels found in {config_path}")

    messages: list[dict] = []
    session_arg = StringSession(string_session) if string_session else str(session_path)
    async with TelegramClient(session_arg, int(api_id), api_hash) as client:
        for channel in channels:
            handle = channel["handle"]
            async for msg in client.iter_messages(handle, limit=limit_per_channel):
                text = msg.message or ""
                if not text.strip():
                    continue
                messages.append(
                    {
                        "channel": handle,
                        "channel_name": channel.get("name", handle),
                        "message_id": msg.id,
                        "date": msg.date.isoformat(),
                        "text": text,
                        "url": f"https://t.me/{handle.lstrip('@')}/{msg.id}" if not str(handle).startswith("http") else None,
                        "region_hint": channel.get("region_hint"),
                        "source_reliability": channel.get("reliability"),
                    }
                )
    return messages
