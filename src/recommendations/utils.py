from typing import Optional

from src.profiles.constants import PROFILE_LABEL_KEY_MAP, SECTOR_LABEL_KEY_MAP

_NORMALIZED_VALUES = set(SECTOR_LABEL_KEY_MAP.values()) | set(
    PROFILE_LABEL_KEY_MAP.values()
)


def normalize_value(value: Optional[str]) -> Optional[str]:
    """
    Normalize a value to its key format.
    Handles both label format ("Startup") and key format ("startup").
    Returns lowercase key format.
    """
    if not value:
        return None

    if value in _NORMALIZED_VALUES:
        return value

    normalized = SECTOR_LABEL_KEY_MAP.get(value) or PROFILE_LABEL_KEY_MAP.get(value)
    if normalized:
        return normalized

    return value.lower().replace(" ", "_").replace("/", "_").replace(",", "")
