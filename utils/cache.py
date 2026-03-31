"""
Per-paper persistent cache with a cross-dataset manifest.

Storage layout:
    results/cache/
        manifest.json                            # global index across all datasets
        {dataset}/{paper_id}/{config_hash}.json  # per-paper result

Config hash is the first 16 hex chars of SHA-256 over canonical JSON of the
rig's config dict (sorted keys). Two calls with identical rig params and model
IDs will hit the same cache entry; any change to params produces a new hash.
"""

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_ROOT = BASE_DIR / "results" / "cache"
MANIFEST_PATH = CACHE_ROOT / "manifest.json"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def build_config_hash(config_dict: dict) -> str:
    """
    Return the first 16 hex chars of SHA-256 over canonical JSON of config_dict.

    config_dict must be JSON-serializable (plain dicts, lists, strings, numbers).
    Raises TypeError if any value is not serializable.
    """
    canonical = json.dumps(config_dict, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-paper cache
# ---------------------------------------------------------------------------

def _paper_cache_path(paper_id: str, config_hash: str, dataset: str) -> Path:
    return CACHE_ROOT / dataset / paper_id / f"{config_hash}.json"


def load_cached(
    paper_id: str,
    config_hash: str,
    dataset: str,
) -> Optional[dict]:
    """
    Return the cached result dict, or None if not present.

    Raises json.JSONDecodeError if the cache file exists but is corrupt.
    """
    path = _paper_cache_path(paper_id, config_hash, dataset)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_result(
    paper_id: str,
    config_hash: str,
    dataset: str,
    data: dict,
) -> None:
    """
    Persist data as JSON. Uses atomic write (write .tmp, then os.replace)
    to avoid corruption on crash.
    """
    path = _paper_cache_path(paper_id, config_hash, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def update_manifest(
    config_hash: str,
    rig_type: str,
    config_summary: str,
    dataset: str,
) -> None:
    """
    Upsert an entry in manifest.json. File-locked for concurrent safety.

    Manifest schema (keyed by config_hash):
    {
      "<hash>": {
        "rig_type": str,
        "config_summary": str,
        "datasets": [str, ...],   # all datasets this config has been used on
        "first_seen": ISO-8601,
        "last_used": ISO-8601,
        "paper_count": int        # total papers cached across all datasets
      }
    }
    """
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = CACHE_ROOT / ".manifest.lock"
    now = datetime.now(timezone.utc).isoformat()

    with lock_path.open("w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            if MANIFEST_PATH.is_file():
                try:
                    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: manifest.json was corrupt; starting fresh.")
                    manifest = {}
            else:
                manifest = {}

            entry = manifest.get(config_hash, {
                "rig_type": rig_type,
                "config_summary": config_summary,
                "datasets": [],
                "first_seen": now,
                "last_used": now,
                "paper_count": 0,
            })
            entry["last_used"] = now
            entry["paper_count"] += 1
            if dataset not in entry["datasets"]:
                entry["datasets"].append(dataset)
            manifest[config_hash] = entry

            tmp = MANIFEST_PATH.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp, MANIFEST_PATH)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def read_manifest() -> dict:
    """Return the full manifest dict, or {} if it does not exist."""
    if not MANIFEST_PATH.is_file():
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)
