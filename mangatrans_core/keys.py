"""API Key management, pool rotation, rate limiting, and system diagnostics for mangatrans."""

import datetime
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

from .config import (
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_MODELS,
    SHARED_COOLDOWNS_FILE,
    SHARED_KEYS_DIR,
    SHARED_KEYS_FILE,
    _bold,
    _c,
    _cyan,
    _dim,
    _green,
    _magenta,
    _red,
    _yellow,
    get_prog_name,
    mask_key,
    pad_visible,
    print_banner,
    strip_ansi,
    visible_width,
)

COOLDOWNS_FILE = SHARED_COOLDOWNS_FILE


class AllKeysExhaustedError(Exception):
    """Raised when all configured API keys have hit rate limits or quota errors."""
    pass


def load_config() -> Dict[str, Any]:
    """Loads JSON configuration from ~/.config/mangatrans/config.json with legacy fallbacks."""
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if isinstance(cfg, dict):
                    if "api_key" in cfg and "keys" not in cfg:
                        old_k = cfg["api_key"].strip()
                        cfg["keys"] = [{
                            "key": old_k,
                            "label": "Default Key",
                            "added_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        }]
                    return cfg
        except Exception:
            pass
    return {"keys": []}


def save_config(cfg: Dict[str, Any]) -> str:
    """Saves configuration securely to disk with atomic write (write-then-rename, 0600 perms)."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp_path = CONFIG_FILE + ".tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp_path, flags, 0o600)
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, CONFIG_FILE)
    return CONFIG_FILE


def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable string like '5h 24m' or '45s'."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def get_seconds_until_gemini_daily_reset() -> float:
    """
    Calculates exact seconds until Google Gemini daily quota resets (Midnight US Pacific Time).
    Adds a 60-second buffer to ensure Google's daily quota bucket has fully rolled over on the servers.
    Falls back to 6 hours only if timezone calculations fail.
    """
    try:
        import zoneinfo
        pt_tz = zoneinfo.ZoneInfo("America/Los_Angeles")
        now_pt = datetime.datetime.now(pt_tz)
        tomorrow_pt = (now_pt + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        secs = (tomorrow_pt - now_pt).total_seconds()
        return max(60.0, secs + 60.0)
    except Exception:
        return 6.0 * 3600.0


def parse_gemini_error_details(http_code: int, headers: Any, body_text: str) -> Tuple[float, str, bool]:
    """
    Analyzes HTTP error code, response headers, and JSON error details to determine:
    1. Exact cooldown duration in seconds (per-minute retry delay or time until Midnight PT).
    2. Detailed human-readable reason description.
    3. Whether the key is permanently invalid / unauthorized.
    """
    is_permanent = False

    err_json = None
    try:
        err_json = json.loads(body_text)
    except Exception:
        pass

    err_status = ""
    err_msg = ""
    retry_delay_from_json = None
    is_daily = False

    if isinstance(err_json, dict) and "error" in err_json:
        err_obj = err_json["error"]
        err_status = str(err_obj.get("status", ""))
        err_msg = str(err_obj.get("message", ""))

        details = err_obj.get("details", [])
        for d in details:
            if not isinstance(d, dict):
                continue
            if "retryDelay" in d:
                rd_str = str(d["retryDelay"]).rstrip("s")
                try:
                    retry_delay_from_json = float(rd_str)
                except ValueError:
                    pass

            meta = d.get("metadata", {})
            q_limit = str(meta.get("quota_limit", ""))
            if any(term in q_limit for term in ["PerDay", "Daily", "per_day", "daily"]):
                is_daily = True

            violations = d.get("violations", [])
            for v in violations:
                desc = str(v.get("description", ""))
                if any(term in desc for term in ["PerDay", "Daily", "per_day", "daily"]):
                    is_daily = True

    retry_delay_from_hdr = None
    if headers:
        ra = headers.get("Retry-After")
        if ra:
            try:
                retry_delay_from_hdr = float(ra)
            except ValueError:
                pass

    if (
        http_code in (400, 403)
        or err_status in ("INVALID_ARGUMENT", "PERMISSION_DENIED", "UNAUTHENTICATED")
        or "API_KEY_INVALID" in body_text
        or "API key not valid" in err_msg
        or "unregistered callers" in err_msg.lower()
        or "caller without identity" in err_msg.lower()
    ):
        is_permanent = True
        cooldown_secs = 86400.0 * 30.0  # 30 days
        reason_desc = f"Invalid/Unregistered key ({err_msg or 'API key rejected by Google'})"
    elif http_code == 429 or "quota" in body_text.lower() or err_status == "RESOURCE_EXHAUSTED":
        if retry_delay_from_json is not None and retry_delay_from_json <= 120:
            cooldown_secs = max(30.0, retry_delay_from_json + 2.0)
            reason_desc = f"Per-minute rate limit (retry in {int(cooldown_secs)}s)"
        elif retry_delay_from_hdr is not None and retry_delay_from_hdr <= 120:
            cooldown_secs = max(30.0, retry_delay_from_hdr + 2.0)
            reason_desc = f"Per-minute rate limit (retry in {int(cooldown_secs)}s)"
        else:
            cooldown_secs = get_seconds_until_gemini_daily_reset()
            reason_desc = f"Daily quota exhausted (resets in {format_duration(cooldown_secs)} at Midnight PT)"
    elif http_code == 404:
        if "unregistered callers" in err_msg.lower() or "identity" in err_msg.lower() or "api key" in err_msg.lower():
            is_permanent = True
            cooldown_secs = 86400.0 * 30.0
            reason_desc = f"Unregistered API Key ({err_msg})"
        else:
            cooldown_secs = 300.0
            reason_desc = f"HTTP 404 ({err_msg or 'Model endpoint not found'})"
    elif http_code in (500, 503):
        cooldown_secs = 60.0
        reason_desc = f"Google temporary server error ({http_code})"
    else:
        cooldown_secs = 300.0
        reason_desc = f"HTTP {http_code} ({err_msg or 'error'})"

    return cooldown_secs, reason_desc, is_permanent


def load_key_cooldowns() -> Dict[str, float]:
    """Loads active rate-limit cooldown timestamps from disk (cross-process & persistent)."""
    try:
        if os.path.isfile(SHARED_COOLDOWNS_FILE):
            with open(SHARED_COOLDOWNS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                now = time.time()
                return {k: float(v) for k, v in data.items() if float(v) > now}
    except Exception:
        pass
    return {}


def save_key_cooldown(key: str, duration_secs: Optional[float] = None) -> None:
    """Saves a key's rate-limit cooldown timestamp to disk (defaults to exact time until Midnight PT)."""
    if duration_secs is None:
        duration_secs = get_seconds_until_gemini_daily_reset()
    try:
        os.makedirs(SHARED_KEYS_DIR, exist_ok=True)
        cooldowns = load_key_cooldowns()
        cooldowns[key] = time.time() + duration_secs
        with open(SHARED_COOLDOWNS_FILE, "w", encoding="utf-8") as f:
            json.dump(cooldowns, f)
    except Exception:
        pass


def clear_key_cooldown(key: str) -> None:
    """Clears cooldown state for a healthy key."""
    try:
        cooldowns = load_key_cooldowns()
        if key in cooldowns:
            del cooldowns[key]
            with open(COOLDOWNS_FILE, "w", encoding="utf-8") as f:
                json.dump(cooldowns, f)
    except Exception:
        pass


def _load_shared_keys() -> List[Dict[str, Any]]:
    """Loads key list from the shared Gemini key pool file (~/.config/gemini-shared/config.json)."""
    if os.path.isfile(SHARED_KEYS_FILE):
        try:
            with open(SHARED_KEYS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data.get("keys", [])
        except Exception:
            pass
    return []


def _save_shared_keys(keys_list: List[Dict[str, Any]]) -> None:
    """Atomically saves key list to the shared Gemini key pool file (0600 perms)."""
    os.makedirs(SHARED_KEYS_DIR, exist_ok=True)
    tmp = SHARED_KEYS_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with open(fd, "w", encoding="utf-8") as f:
        json.dump({"keys": keys_list}, f, indent=2, ensure_ascii=False)
    os.replace(tmp, SHARED_KEYS_FILE)


def collect_all_configured_keys() -> List[Dict[str, str]]:
    """Loads all available keys from environment variables, shared key pool, and local config."""
    pool: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for env_var in ["MANGATRANS_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        val = os.environ.get(env_var)
        if val and val.strip() and val.strip() not in seen:
            pool.append({"key": val.strip(), "label": f"Env var (${env_var})"})
            seen.add(val.strip())

    for item in _load_shared_keys():
        k = item.get("key", "").strip() if isinstance(item, dict) else str(item).strip()
        label = item.get("label", "") if isinstance(item, dict) else ""
        if k and k not in seen:
            pool.append({"key": k, "label": label or f"Key #{len(pool)+1}"})
            seen.add(k)

    cfg = load_config()
    for item in cfg.get("keys", []):
        k = item.get("key", "").strip() if isinstance(item, dict) else str(item).strip()
        label = item.get("label", "") if isinstance(item, dict) else ""
        if k and k not in seen:
            pool.append({"key": k, "label": label or f"Key #{len(pool)+1}"})
            seen.add(k)

    if not pool and cfg.get("api_key"):
        k = cfg["api_key"].strip()
        if k and k not in seen:
            pool.append({"key": k, "label": "Configured Key"})
            seen.add(k)

    return pool


def resolve_key_index(target: str, keys_list: List[Dict[str, Any]]) -> Optional[int]:
    """Resolves target key by serial number (#), unique label, masked key, or exact API key."""
    if not target or not keys_list:
        return None
    target_clean = target.strip()

    if target_clean.isdigit():
        idx = int(target_clean) - 1
        if 0 <= idx < len(keys_list):
            return idx

    for i, item in enumerate(keys_list):
        lbl = item.get("label", "").strip()
        if lbl.lower() == target_clean.lower():
            return i

    for i, item in enumerate(keys_list):
        k = item.get("key", "").strip()
        if k == target_clean or target_clean in mask_key(k):
            return i

    for i, item in enumerate(keys_list):
        lbl = item.get("label", "").strip()
        if target_clean.lower() in lbl.lower():
            return i

    return None


def parse_key_selection(
    spec: Optional[Union[str, List[str]]],
    available_keys: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Parses key selection specifier into a list of key dicts (maintaining user-specified sequence).
    Supports 1-based indices ('1', '3'), ranges ('1-3', '2-5'), labels ('account1', 'gmail'),
    masked keys, or raw API keys.
    """
    if not spec:
        return []

    candidate_tokens: List[str] = []
    if isinstance(spec, str):
        candidate_tokens.append(spec)
    elif isinstance(spec, (list, tuple)):
        for item in spec:
            if isinstance(item, str):
                candidate_tokens.append(item)

    comma_tokens: List[str] = []
    for ct in candidate_tokens:
        for piece in ct.split(","):
            p = piece.strip()
            if p:
                comma_tokens.append(p)

    final_tokens: List[str] = []
    for tok in comma_tokens:
        if " " in tok or "\t" in tok:
            if resolve_key_index(tok, available_keys) is not None:
                final_tokens.append(tok)
            else:
                parts = [p.strip() for p in re.split(r"\s+", tok) if p.strip()]
                final_tokens.extend(parts)
        else:
            final_tokens.append(tok)

    selected: List[Dict[str, str]] = []
    seen_keys: Set[str] = set()

    def _add(entry: Dict[str, str]) -> None:
        k = entry["key"]
        if k not in seen_keys:
            selected.append(entry)
            seen_keys.add(k)

    total_avail = len(available_keys)

    for tok in final_tokens:
        m_range = re.match(r"^(\d+)-(\d+)$", tok)
        if m_range:
            a, b = int(m_range.group(1)), int(m_range.group(2))
            lo, hi = min(a, b), max(a, b)
            matched_any = False
            for idx in range(lo, hi + 1):
                if 1 <= idx <= total_avail:
                    _add(available_keys[idx - 1])
                    matched_any = True
                else:
                    sys.stderr.write(_yellow(f"  ⚠ Key index #{idx} is out of range (total: {total_avail}), skipped.\n"))
            if not matched_any:
                sys.stderr.write(_yellow(f"  ⚠ Key range '{tok}' matched no available keys.\n"))
            continue

        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < total_avail:
                _add(available_keys[idx])
            else:
                sys.stderr.write(_yellow(f"  ⚠ Key #{tok} is out of range (total: {total_avail}), skipped.\n"))
            continue

        matched_idx = resolve_key_index(tok, available_keys)
        if matched_idx is not None:
            _add(available_keys[matched_idx])
            continue

        if len(tok) >= 20 or tok.startswith(("AIza", "AQ.")):
            _add({
                "key": tok,
                "label": f"CLI argument ({mask_key(tok)})",
            })
            continue

        sys.stderr.write(_yellow(f"  ⚠ Key specifier '{tok}' not recognized as a valid index, label, or API key.\n"))

    return selected


class DynamicKeyManager:
    """
    Manages API keys with:
    1. Sequential rotation across full pool or explicit subset.
    2. Explicit key selection: accepts 1-based indices (e.g. 1, 1-3), labels, or raw keys.
    3. Dynamic config file reloading: detects newly added keys mid-translation.
    4. Persistent cross-process cooldown cache.
    5. Smart error classification.
    6. Graceful termination on full pool exhaustion with partial save and resume support.
    """
    def __init__(self, explicit_keys: Optional[Union[str, List[str]]] = None):
        self.explicit_keys = explicit_keys
        self.is_explicit_selection = bool(explicit_keys)
        self.pool: List[Dict[str, str]] = []
        self.current_idx: int = 0
        self.exhausted_keys: Set[str] = set()
        self.permanently_failed: Set[str] = set()
        self.last_config_mtime: float = 0.0
        self.reload_keys(initial=True)

    def reload_keys(self, initial: bool = False) -> bool:
        """Checks for keys from CLI selection or reloads from environment and config.json."""
        if self.is_explicit_selection:
            if initial:
                all_avail = collect_all_configured_keys()
                self.pool = parse_key_selection(self.explicit_keys, all_avail)
            return False

        try:
            mtime = max(
                os.path.getmtime(CONFIG_FILE) if os.path.isfile(CONFIG_FILE) else 0.0,
                os.path.getmtime(SHARED_KEYS_FILE) if os.path.isfile(SHARED_KEYS_FILE) else 0.0,
            )
        except Exception:
            mtime = 0.0

        if not initial and mtime <= self.last_config_mtime:
            return False

        old_keys = {item["key"] for item in self.pool}
        new_pool = collect_all_configured_keys()

        added = [item for item in new_pool if item["key"] not in old_keys]
        self.pool = new_pool
        self.last_config_mtime = mtime

        if added and not initial:
            for item in added:
                self.exhausted_keys.discard(item["key"])
                self.permanently_failed.discard(item["key"])
                clear_key_cooldown(item["key"])
            sys.stdout.write(f"\n    ✔ Dynamic config update detected: {len(added)} new API key(s) added to pool.\n")
            sys.stdout.flush()
            return True

        return False

    def get_current_key(self) -> Optional[Dict[str, str]]:
        """Returns the first non-exhausted, non-cooldown key entry."""
        self.reload_keys()
        if not self.pool:
            return None
        cooldowns = load_key_cooldowns()
        now = time.time()
        for offset in range(len(self.pool)):
            idx = (self.current_idx + offset) % len(self.pool)
            k = self.pool[idx]["key"]
            if k not in self.exhausted_keys and k not in self.permanently_failed and now >= cooldowns.get(k, 0.0):
                self.current_idx = idx
                return self.pool[idx]
        return None

    def rotate_to_next_key(
        self,
        failed_key: str,
        reason: str = "Rate limit / 429",
        cooldown_secs: Optional[float] = None,
        is_permanent: bool = False,
        progress_listener: Optional[Any] = None,
    ) -> Tuple[Optional[Dict[str, str]], bool]:
        """
        Marks failed key into persistent cooldown, re-checks config for new keys,
        and advances to the next available healthy key.
        Returns (next_key_info, pool_exhausted).
        """
        if cooldown_secs is None:
            cooldown_secs = get_seconds_until_gemini_daily_reset()
        self.exhausted_keys.add(failed_key)
        if is_permanent:
            self.permanently_failed.add(failed_key)
        save_key_cooldown(failed_key, duration_secs=cooldown_secs)
        self.reload_keys()

        cooldowns = load_key_cooldowns()
        now = time.time()

        for offset in range(len(self.pool)):
            idx = (self.current_idx + 1 + offset) % len(self.pool)
            candidate = self.pool[idx]
            k = candidate["key"]
            if k not in self.exhausted_keys and k not in self.permanently_failed and now >= cooldowns.get(k, 0.0):
                self.current_idx = idx
                return candidate, False

        self.last_config_mtime = 0.0
        if self.reload_keys():
            cooldowns = load_key_cooldowns()
            for idx, candidate in enumerate(self.pool):
                k = candidate["key"]
                if k not in self.exhausted_keys and k not in self.permanently_failed and now >= cooldowns.get(k, 0.0):
                    self.current_idx = idx
                    return candidate, False

        eligible_keys = [c for c in self.pool if c["key"] not in self.permanently_failed]
        if eligible_keys:
            earliest_time = min(cooldowns.get(c["key"], now + 999999) for c in eligible_keys)
            wait_needed = max(1.0, earliest_time - now)
            if wait_needed <= 60.0:
                msg = f"    ⏳ All available keys temporarily cooling down. Waiting {int(wait_needed)}s for rate-limit reset..."
                if progress_listener and hasattr(progress_listener, "print_notice"):
                    progress_listener.print_notice(msg)
                else:
                    sys.stdout.write(f"\n{msg}\n")
                    sys.stdout.flush()
                time.sleep(wait_needed + 1.0)
                self.exhausted_keys.clear()
                cooldowns2 = load_key_cooldowns()
                now2 = time.time()
                for candidate in eligible_keys:
                    k2 = candidate["key"]
                    if k2 not in self.permanently_failed and now2 >= cooldowns2.get(k2, 0.0):
                        try:
                            self.current_idx = self.pool.index(candidate)
                        except ValueError:
                            self.current_idx = 0
                        return candidate, False
                self.current_idx = 0
                return eligible_keys[0], False

        return None, True

    def reset_cycle(self) -> None:
        """Resets key exhaustion for a new translation run or retry."""
        self.exhausted_keys.clear()

    def get_key_status_str(self) -> str:
        """Returns a compact '[N/M: label]' display string for the current key."""
        key_info = self.get_current_key()
        if not key_info:
            return "[no active key]"
        cooldowns = load_key_cooldowns()
        now = time.time()
        session_usable = [
            k for k in self.pool
            if k["key"] not in self.permanently_failed
            and (cooldowns.get(k["key"], 0.0) - now) < 86400
        ]
        try:
            n = next(i + 1 for i, k in enumerate(session_usable) if k["key"] == key_info["key"])
        except StopIteration:
            n = "?"
        m = len(session_usable)
        label = key_info.get("label", mask_key(key_info.get("key", "")))
        return f"[{n}/{m}] {_dim(label)}"


def get_key_pool(explicit_keys: Optional[Union[str, List[str]]] = None) -> List[Dict[str, str]]:
    """Returns static snapshot of current key pool."""
    mgr = DynamicKeyManager(explicit_keys)
    return mgr.pool


def format_datetime(iso_str: str) -> str:
    """Formats an ISO-8601 timestamp string into local YYYY-MM-DD HH:MM."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.datetime.fromisoformat(iso_str)
        local_dt = dt.astimezone()
        return local_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        if len(iso_str) >= 16:
            return iso_str[:16].replace("T", " ")
        return iso_str[:10] if len(iso_str) >= 10 else (iso_str or "—")


def format_status(k: str, cd_time: float, now: float) -> str:
    """Formats key status with exact countdown, timestamp, and ANSI colors."""
    if cd_time > now:
        diff = cd_time - now
        dt = datetime.datetime.fromtimestamp(cd_time)
        now_dt = datetime.datetime.fromtimestamp(now)
        if dt.date() == now_dt.date():
            ts_str = f"today {dt.strftime('%H:%M')}"
        elif dt.date() == (now_dt + datetime.timedelta(days=1)).date():
            ts_str = f"tomorrow {dt.strftime('%H:%M')}"
        else:
            ts_str = dt.strftime("%Y-%m-%d %H:%M")
        if diff > 86400:
            return _red(f"✘ Banned (~{format_duration(diff)})")
        return _yellow(f"⏳ Cooldown ({format_duration(diff)} → {ts_str})")
    return _green("✔ Ready")


def add_or_update_key(key: str, label: Optional[str] = None) -> str:
    """Adds a new key or updates an existing key in the persistent pool with mandatory unique label."""
    key = key.strip()
    if not key:
        return "✘ Error: API key cannot be empty."

    keys_list = _load_shared_keys()

    if not label or not label.strip():
        return f"✘ Error: A unique label is required (e.g. {get_prog_name()} --add-key <KEY> --label \"account_email\")."

    label = label.strip()

    for i, item in enumerate(keys_list):
        k = item.get("key", "").strip() if isinstance(item, dict) else str(item).strip()
        lbl = item.get("label", "").strip() if isinstance(item, dict) else ""
        if lbl.lower() == label.lower() and k != key:
            return f"✘ Error: Label '{label}' is already in use by key #{i+1} ({mask_key(k)}). Labels must be unique."

    existing_idx = None
    for i, item in enumerate(keys_list):
        k = item.get("key", "").strip() if isinstance(item, dict) else str(item).strip()
        if k == key:
            existing_idx = i
            break

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if existing_idx is not None:
        old_lbl = keys_list[existing_idx].get("label", "")
        keys_list[existing_idx]["label"] = label
        keys_list[existing_idx]["updated_at"] = now_iso
        msg = f"✔ Updated key #{existing_idx + 1} ({mask_key(key)}) label: '{old_lbl}' → '{label}'."
    else:
        entry = {
            "key": key,
            "label": label,
            "added_at": now_iso,
        }
        keys_list.append(entry)
        msg = f"✔ Added new API key #{len(keys_list)} ({mask_key(key)}) with unique label: '{label}'."

    _save_shared_keys(keys_list)
    return msg


def update_key_entry(target: str, new_key: Optional[str] = None, new_label: Optional[str] = None) -> bool:
    """Updates an API key or its unique label identified by serial number (#) or label in the shared pool."""
    keys_list = _load_shared_keys()
    if not keys_list:
        print("✘ Error: No API keys configured in shared pool.", file=sys.stderr)
        return False

    idx = resolve_key_index(target, keys_list)
    if idx is None:
        print(f"✘ Error: Key '{target}' not found by # index or label.", file=sys.stderr)
        return False

    item = keys_list[idx]
    old_key = item.get("key", "")
    old_label = item.get("label", "")
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if new_label is not None:
        new_label = new_label.strip()
        if not new_label:
            print("✘ Error: Label cannot be empty.", file=sys.stderr)
            return False
        for i, other in enumerate(keys_list):
            if i != idx and other.get("label", "").strip().lower() == new_label.lower():
                print(f"✘ Error: Label '{new_label}' is already in use by key #{i+1} ({mask_key(other.get('key', ''))}).", file=sys.stderr)
                return False
        item["label"] = new_label

    if new_key is not None:
        new_key = new_key.strip()
        if not new_key:
            print("✘ Error: API key cannot be empty.", file=sys.stderr)
            return False
        item["key"] = new_key
        clear_key_cooldown(old_key)

    item["updated_at"] = now_iso
    keys_list[idx] = item
    _save_shared_keys(keys_list)
    print(f"✔ Successfully updated key #{idx + 1}:")
    print(f"    Label: '{old_label}' → '{item['label']}'")
    print(f"    Key:   {mask_key(old_key)} → {mask_key(item['key'])}")
    return True


def remove_key(target: str) -> bool:
    """Removes a key by numeric 1-based serial number (#), unique label match, or key string from the shared pool."""
    keys_list = _load_shared_keys()
    if not keys_list:
        print("✘ Error: No API keys configured in shared pool.", file=sys.stderr)
        return False

    idx_to_remove = resolve_key_index(target, keys_list)
    if idx_to_remove is not None:
        removed = keys_list.pop(idx_to_remove)
        _save_shared_keys(keys_list)
        clear_key_cooldown(removed.get("key", ""))
        print(f"✔ Removed key #{idx_to_remove + 1} ({mask_key(removed.get('key', ''))} — '{removed.get('label', '')}').")
        return True

    print(f"✘ Error: Key '{target}' not found by # index or label.", file=sys.stderr)
    return False


def reset_key_cooldown(target: str = "all") -> bool:
    """Resets cooldown timestamp for a specific key (by # or label) or all keys."""
    if target.lower() in ("all", "pool", "*"):
        try:
            if os.path.isfile(SHARED_COOLDOWNS_FILE):
                os.remove(SHARED_COOLDOWNS_FILE)
            print("✔ Cleared all API key rate-limit cooldowns. All keys are now Ready.")
            return True
        except Exception as e:
            print(f"✘ Error clearing cooldowns: {e}", file=sys.stderr)
            return False

    keys_list = _load_shared_keys()
    idx = resolve_key_index(target, keys_list)
    if idx is None:
        print(f"✘ Error: Key '{target}' not found.", file=sys.stderr)
        return False

    k = keys_list[idx].get("key", "")
    lbl = keys_list[idx].get("label", "")
    clear_key_cooldown(k)
    print(f"✔ Cleared cooldown for key #{idx + 1} ({mask_key(k)} — '{lbl}'). Status: Ready.")
    return True


def test_key_health(target: str = "all") -> None:
    """Tests health and accessibility of configured API keys by making lightweight validation pings with real-time streaming output."""
    keys_list = _load_shared_keys()
    if not keys_list:
        print("✘ Error: No API keys configured in shared pool.", file=sys.stderr)
        return

    if target.lower() not in ("all", "pool", "*"):
        idx = resolve_key_index(target, keys_list)
        if idx is None:
            print(f"✘ Error: Key '{target}' not found.", file=sys.stderr)
            return
        keys_to_test = [(idx, keys_list[idx])]
    else:
        keys_to_test = list(enumerate(keys_list))

    print_banner("API Key Health Diagnostics")

    headers = ("#", "API Key", "Label / Account / Note", "Diagnostic Result / Status")
    col_w = [
        max(visible_width(headers[0]), max(visible_width(str(i + 1)) for i, _ in keys_to_test)),
        max(visible_width(headers[1]), max(visible_width(mask_key(item.get("key", ""))) for _, item in keys_to_test)),
        max(visible_width(headers[2]), max(visible_width(item.get("label", f"Key #{i+1}")) for i, item in keys_to_test)),
        visible_width(headers[3]),
    ]

    header_line = "  " + "  ".join(pad_visible(headers[j], col_w[j]) for j in range(4))
    sep_line = "  " + "  ".join("-" * col_w[j] for j in range(4))
    print(header_line)
    print(sep_line)
    sys.stdout.flush()

    for i, item in keys_to_test:
        k = item.get("key", "")
        lbl = item.get("label", f"Key #{i+1}")

        diag_status = "✔ Healthy & Active"
        payload = {
            "contents": [{"parts": [{"text": "ping"}]}],
            "generationConfig": {"maxOutputTokens": 1},
        }
        encoded = json.dumps(payload).encode("utf-8")

        for model in DEFAULT_MODELS:
            clean_model = model.replace("models/", "")
            test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"
            req = urllib.request.Request(
                test_url, data=encoded,
                headers={"Content-Type": "application/json", "x-goog-api-key": k}
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        clear_key_cooldown(k)
                        diag_status = f"✔ Healthy & Active ({clean_model})"
                        break
                    else:
                        diag_status = f"⚠ HTTP {resp.status}"
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                cd_secs, desc, is_perm = parse_gemini_error_details(e.code, e.headers, body)

                if (
                    e.code in (500, 502, 503, 504)
                    or (e.code in (404, 400) and not is_perm)
                ):
                    diag_status = f"⚠ HTTP {e.code} ({desc})"
                    continue

                save_key_cooldown(k, duration_secs=cd_secs)
                if is_perm:
                    diag_status = f"✘ Denied / Invalid ({desc})"
                elif e.code == 429 or "quota" in body.lower():
                    diag_status = f"⏳ Quota / Rate Limited ({desc})"
                else:
                    diag_status = f"⚠ HTTP {e.code} ({desc})"
                break
            except Exception:
                diag_status = f"⚠ Timeout / Error ({clean_model})"
                continue

        row = (str(i + 1), mask_key(k), lbl, diag_status)
        line = "  " + "  ".join(pad_visible(row[j], col_w[j]) for j in range(4))
        print(line)
        sys.stdout.flush()
    print()


def list_keys() -> None:
    """Displays a clean tabular list of all keys in the shared Gemini pool with exact timestamps and robust visual alignment."""
    keys_list = _load_shared_keys()
    cooldowns = load_key_cooldowns()
    now = time.time()
    print_banner("Shared Gemini API Key Pool (~/.config/gemini-shared/config.json)")
    if not keys_list:
        print("  No API keys configured in the shared pool.")
        print(f"  Add one with: {get_prog_name()} --add-key <KEY> --label \"unique_label\"")
    else:
        headers = ("#", "API Key", "Label", "Date Added", "Status")
        rows = []
        for i, item in enumerate(keys_list):
            k = item.get("key", "")
            lbl = item.get("label", f"Account #{i+1}")
            added = format_datetime(item.get("added_at", ""))
            cd_time = cooldowns.get(k, 0.0)
            status = format_status(k, cd_time, now)
            rows.append((str(i + 1), mask_key(k), lbl, added, status))

        num_cols = len(headers)
        col_w = [
            max(visible_width(headers[j]), max(visible_width(r[j]) for r in rows))
            for j in range(num_cols)
        ]

        header_line = "  " + "  ".join(pad_visible(headers[j], col_w[j]) for j in range(num_cols))
        sep_line = "  " + "  ".join("-" * col_w[j] for j in range(num_cols))
        print(header_line)
        print(sep_line)
        for r in rows:
            padded_cells = []
            for j in range(num_cols):
                align = "right" if j == 0 else "left"
                padded_cells.append(pad_visible(r[j], col_w[j], align=align))
            print("  " + "  ".join(padded_cells))

    print(f"\n  Commands:")
    print(f"    Add key:            {get_prog_name()} --add-key <KEY> --label \"unique_label\"")
    print(f"    Update key/label:   {get_prog_name()} --update-key <# | LABEL> [--new-label <NAME>] [--new-key <KEY>]")
    print(f"    Remove key:         {get_prog_name()} --remove-key <# | LABEL>")
    print(f"    Reset cooldown:     {get_prog_name()} --reset-cooldown <# | LABEL | all>")
    print(f"    Test key health:    {get_prog_name()} --test-key <# | LABEL | all>")


def check_deps() -> None:
    """Inspects Python environment, imaging libraries, CLI tools, and API health."""
    print_banner("System & Dependency Status")
    rows = []

    py_ver = sys.version.split()[0]
    rows.append(("Python", f"✔  installed ({py_ver})", "Core runtime environment"))

    if Image:
        import PIL
        rows.append(("Pillow", f"✔  installed ({getattr(PIL, '__version__', 'ok')})", "Image processing & raster engine"))
    else:
        rows.append(("Pillow", "✘  missing", "Run: pip install Pillow"))

    if cv2:
        rows.append(("OpenCV", f"✔  installed ({cv2.__version__})", "Inpainting & vision algorithms"))
    else:
        rows.append(("OpenCV", "✘  missing", "Run: pip install opencv-python"))

    if np:
        rows.append(("NumPy", f"✔  installed ({np.__version__})", "Numerical array manipulations"))
    else:
        rows.append(("NumPy", "✘  missing", "Run: pip install numpy"))

    for tool, desc in [
        ("ffmpeg", "PDF/image conversion pipeline"),
        ("pdftoppm", "High-speed PDF page rasterizer"),
        ("unrar", "CBR archive extraction"),
        ("7z", "7-Zip / CBR archive extraction fallback"),
    ]:
        p = shutil.which(tool)
        if p:
            rows.append((tool, "✔  installed", p))
        else:
            rows.append((tool, "ℹ  optional", f"{desc} (install via apt)"))

    pool = get_key_pool()
    if pool:
        active_key = pool[0]["key"]
        test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_MODELS[0]}"
        req = urllib.request.Request(test_url, headers={"x-goog-api-key": active_key})
        try:
            with urllib.request.urlopen(req, timeout=7) as resp:
                if resp.status == 200:
                    rows.append(("Google AI API", f"✔  active ({mask_key(active_key)})", f"{len(pool)} key(s) in pool"))
                else:
                    rows.append(("Google AI API", f"⚠  HTTP {resp.status}", "Unexpected response status"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            _, desc, _ = parse_gemini_error_details(e.code, e.headers, body)
            rows.append(("Google AI API", f"⚠  HTTP {e.code}", f"{desc} ({mask_key(active_key)})"))
        except Exception as e:
            rows.append(("Google AI API", "⚠  unreachable", f"Error: {e}"))
    else:
        rows.append(("Google AI API", "✘  missing", f"Add an API key with: {get_prog_name()} --add-key <KEY> --label \"unique_label\""))

    col_w = [
        max(visible_width(r[0]) for r in rows),
        max(visible_width(r[1]) for r in rows),
        max(visible_width(r[2]) for r in rows),
    ]

    for r in rows:
        print(f"  {pad_visible(r[0], col_w[0])}  {pad_visible(r[1], col_w[1])}  {pad_visible(r[2], col_w[2])}")
    print()
