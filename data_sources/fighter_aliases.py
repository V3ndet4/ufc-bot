from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_ALIAS_OVERRIDE_COLUMNS = ["source_name", "canonical_name", "notes"]


def normalize_fighter_name(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_only = ascii_only.lower().replace("’", "'").replace("`", "'")
    ascii_only = re.sub(r"\b(jr|jr\.|sr|sr\.|iii|iv|v)\b", " ", ascii_only)
    ascii_only = re.sub(r"[^a-z0-9']+", " ", ascii_only)
    return " ".join(ascii_only.split())


def load_fighter_alias_overrides(path: str | Path) -> pd.DataFrame:
    alias_path = Path(path)
    if not alias_path.exists():
        return pd.DataFrame(columns=DEFAULT_ALIAS_OVERRIDE_COLUMNS)

    loaded = pd.read_csv(alias_path)
    if loaded.empty:
        return pd.DataFrame(columns=DEFAULT_ALIAS_OVERRIDE_COLUMNS)

    normalized = loaded.copy()
    for column in DEFAULT_ALIAS_OVERRIDE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()

    normalized = normalized.loc[
        normalized["source_name"].ne("") & normalized["canonical_name"].ne(""),
        DEFAULT_ALIAS_OVERRIDE_COLUMNS,
    ].copy()
    if normalized.empty:
        return pd.DataFrame(columns=DEFAULT_ALIAS_OVERRIDE_COLUMNS)
    return normalized.reset_index(drop=True)


def build_fighter_alias_lookup(alias_overrides: pd.DataFrame | None) -> dict[str, str]:
    if alias_overrides is None or alias_overrides.empty:
        return {}

    lookup: dict[str, str] = {}
    for row in alias_overrides.to_dict(orient="records"):
        source_name = str(row.get("source_name", "") or "").strip()
        canonical_name = str(row.get("canonical_name", "") or "").strip()
        source_key = normalize_fighter_name(source_name)
        if not source_key or not canonical_name:
            continue
        lookup[source_key] = canonical_name
    return lookup


def resolve_fighter_alias(value: object, alias_lookup: dict[str, str] | None = None) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    if not alias_lookup:
        return cleaned
    return str(alias_lookup.get(normalize_fighter_name(cleaned), cleaned)).strip()


def fighter_alias_key(value: object, alias_lookup: dict[str, str] | None = None) -> str:
    return normalize_fighter_name(resolve_fighter_alias(value, alias_lookup))
