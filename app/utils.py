from pathlib import Path


def check_source_exists(source: str) -> bool:
    if source.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return True

    path = Path(source)
    return path.exists() and path.is_file()
