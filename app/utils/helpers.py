"""Utility helper functions."""

import hashlib
import re
from datetime import datetime
from typing import Any, Optional
from uuid import UUID


def generate_hash(data: str, length: int = 16) -> str:
    """Generate a SHA256 hash of data.

    Args:
        data: String to hash
        length: Length of returned hash (max 64)

    Returns:
        Hex string of specified length
    """
    return hashlib.sha256(data.encode()).hexdigest()[:length]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to specified length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean and normalize text.

    Removes excessive whitespace and special characters.
    """
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep punctuation
    text = re.sub(r"[^\w\s.,!?;:\-'\"()\[\]{}]", "", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple word-based approximation (actual count varies by model).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    # Rough approximation: ~1.3 tokens per word for English
    words = len(text.split())
    return int(words * 1.3)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def parse_uuid(value: Any) -> Optional[UUID]:
    """Safely parse a value to UUID.

    Args:
        value: Value to parse (string or UUID)

    Returns:
        UUID or None if invalid
    """
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            return None
    return None


def is_valid_uuid(value: str) -> bool:
    """Check if string is a valid UUID.

    Args:
        value: String to check

    Returns:
        True if valid UUID
    """
    try:
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def datetime_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO format string.

    Args:
        dt: Datetime object

    Returns:
        ISO format string
    """
    return dt.isoformat()


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely parse JSON string.

    Args:
        data: JSON string
        default: Default value if parsing fails

    Returns:
        Parsed data or default
    """
    import json

    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split list into chunks.

    Args:
        lst: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: dict) -> dict:
    """Merge multiple dictionaries.

    Later dicts override earlier ones.

    Args:
        dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    # Remove special characters
    filename = re.sub(r"[^\w\-_.]", "_", filename)
    # Remove multiple underscores
    filename = re.sub(r"_+", "_", filename)
    # Limit length
    if len(filename) > 200:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        filename = name[:200 - len(ext) - 1] + "." + ext if ext else name[:200]
    return filename
