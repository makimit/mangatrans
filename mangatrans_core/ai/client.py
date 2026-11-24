"""Gemini API client with sequential key rotation, model cascading, and structured JSON parsing."""

import base64
import json
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from ..config import (
    DEFAULT_MODELS,
    LiveProgress,
    _bold,
    _cyan,
    _yellow,
    detect_image_mime_type,
    format_duration,
    mask_key,
)
from ..keys import (
    AllKeysExhaustedError,
    DynamicKeyManager,
    clear_key_cooldown,
    parse_gemini_error_details,
)
from .prompt import build_translate_prompt


def sanitize_gemini_items(items: Any) -> List[Dict[str, Any]]:
    """Guarantees every item in the JSON response has normalized string text and valid schema types."""
    if not isinstance(items, list):
        return []
    clean = []
    for it in items:
        if not isinstance(it, dict):
            continue
        trans = it.get("translated_text", "")
        if isinstance(trans, list):
            trans = "\n".join(str(x) for x in trans)
        elif trans is None:
            trans = ""
        else:
            trans = str(trans)

        orig = it.get("original_text", "")
        if isinstance(orig, list):
            orig = "\n".join(str(x) for x in orig)
        elif orig is None:
            orig = ""
        else:
            orig = str(orig)

        b_type = str(it.get("bubble_type") or "dialogue").strip().lower()
        c_type = str(it.get("container_type") or "bubble").strip().lower()
        bg_col = str(it.get("bg_color") or "white").strip().lower()
        box = it.get("box_2d", [])
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            box = [int(v) for v in box]
        except (ValueError, TypeError):
            continue

        clean.append({
            "box_2d": box,
            "original_text": orig.strip(),
            "translated_text": trans.strip(),
            "bubble_type": b_type,
            "container_type": c_type,
            "bg_color": bg_col,
        })
    return clean


def parse_gemini_json(raw: str) -> Dict[str, Any]:
    """Robustly extracts and parses JSON payload from Gemini response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    data = None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            data = parsed
        elif isinstance(parsed, list):
            data = {"items": parsed}
    except Exception:
        pass
    if data is None:
        first_b = raw.find("{")
        last_b = raw.rfind("}")
        if first_b != -1 and last_b != -1 and last_b > first_b:
            try:
                parsed = json.loads(raw[first_b:last_b + 1])
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                pass
    if data is None:
        return {"items": []}

    data["items"] = sanitize_gemini_items(data.get("items", []))
    return data


def _call_gemini_api(
    payload_dict: Dict[str, Any],
    key_manager: DynamicKeyManager,
    label: str = "API",
    progress: Optional[LiveProgress] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Executes a Gemini API request with sequential 1 -> 2 -> ... -> N key cycling,
    dynamic key pool reloading from config.json, intelligent rate-limit detection, and model cascading.
    """
    encoded = json.dumps(payload_dict).encode("utf-8")

    while True:
        key_info = key_manager.get_current_key()
        if not key_info:
            return None, None, None

        api_key = key_info["key"]
        key_label = key_info.get("label", mask_key(api_key))

        last_error_desc = "API Error"
        last_cooldown_secs = None
        last_is_perm = False

        for idx, model in enumerate(DEFAULT_MODELS):
            clean_model = model.replace("models/", "")
            next_model = (
                DEFAULT_MODELS[idx + 1].replace("models/", "")
                if idx + 1 < len(DEFAULT_MODELS)
                else None
            )

            if progress:
                key_disp = key_manager.get_key_status_str()
                base_text = getattr(progress, "base_label", progress.message)
                m_match = re.match(
                    r"^(\s*\[\d+/\d+\]\s*)(?:\[\d+/\d+\]\s*.*?Processing|\s*Processing)\s+(.+?)\.\.\.",
                    base_text,
                )
                if m_match:
                    p_prefix = m_match.group(1).strip()
                    p_target = m_match.group(2).strip()
                    progress.set_message(
                        f"{p_prefix} {key_disp} ({_cyan(clean_model)})  Processing {p_target}..."
                    )
                else:
                    progress.set_message(f"{key_disp} ({_cyan(clean_model)})  {label}...")

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }
            req = urllib.request.Request(url, data=encoded, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    clear_key_cooldown(api_key)
                    return res_json, key_label, clean_model
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                cd_secs, desc, is_perm = parse_gemini_error_details(e.code, e.headers, body)
                last_error_desc = desc
                last_cooldown_secs = cd_secs
                last_is_perm = is_perm

                if (
                    e.code in (500, 502, 503, 504)
                    or (e.code in (404, 400) and not is_perm)
                ):
                    fallback_info = (
                        f"falling back to '{next_model}'..."
                        if next_model
                        else "no more fallback models on this key."
                    )
                    notice = f"    ↻ Model '{clean_model}' returned HTTP {e.code} ({desc}) -> {fallback_info}"
                    if progress:
                        progress.print_notice(_yellow(notice))
                    else:
                        sys.stderr.write(f"\n{_yellow(notice)}\n")
                        sys.stderr.flush()
                    continue

                notice = f"    ↻ Key '{key_label}' {desc} -> cooling down for {format_duration(cd_secs)}. Switching key..."
                if progress:
                    progress.print_notice(_yellow(notice))
                else:
                    sys.stderr.write(f"\n{_yellow(notice)}\n")
                    sys.stderr.flush()
                break
            except (TimeoutError, urllib.error.URLError, Exception) as e:
                is_timeout = isinstance(e, TimeoutError) or (
                    isinstance(e, urllib.error.URLError)
                    and (
                        "timed out" in str(e).lower()
                        or isinstance(getattr(e, "reason", None), TimeoutError)
                    )
                )
                if is_timeout:
                    err_msg = "timed out after 30s"
                else:
                    err_msg = f"network error ({getattr(e, 'reason', e)})"

                last_error_desc = err_msg
                last_cooldown_secs = 60.0
                fallback_info = (
                    f"falling back to '{next_model}'..."
                    if next_model
                    else "no more fallback models on this key."
                )
                notice = f"    ↻ Model '{clean_model}' {err_msg} -> {fallback_info}"
                if progress:
                    progress.print_notice(_yellow(notice))
                else:
                    sys.stderr.write(f"\n{_yellow(notice)}\n")
                    sys.stderr.flush()
                continue

        next_key, all_exhausted = key_manager.rotate_to_next_key(
            api_key,
            reason=last_error_desc,
            cooldown_secs=last_cooldown_secs,
            is_permanent=last_is_perm,
            progress_listener=progress,
        )
        if all_exhausted:
            return None, None, None
        if next_key:
            new_disp = key_manager.get_key_status_str()
            new_notice = f"    {_cyan('→')} Now using key {_bold(new_disp)}"
            if progress:
                progress.print_notice(new_notice)
            else:
                sys.stderr.write(f"\n{new_notice}\n")
                sys.stderr.flush()


def query_page_translation(
    image_bytes: bytes,
    key_manager: DynamicKeyManager,
    src_lang_name: str = "Japanese",
    tgt_lang_name: str = "English",
    src_lang_code: str = "ja",
    narrative_context: str = "",
    progress: Optional[LiveProgress] = None,
) -> Dict[str, Any]:
    """Sends page image bytes to Gemini vision for structured translation JSON."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = detect_image_mime_type(image_bytes)
    prompt = build_translate_prompt(
        src_lang_name, tgt_lang_name, src_lang_code, narrative_context
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": b64}},
            ]
        }],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ],
    }

    data, key_label, model = _call_gemini_api(payload, key_manager, "API", progress=progress)
    if data is None:
        raise AllKeysExhaustedError("All configured API keys have hit their rate/quota limits.")

    try:
        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        result = parse_gemini_json(raw)
        if isinstance(result, dict) and "items" in result:
            result["_model_used"] = model
            return result
        return {"items": [], "_model_used": model}
    except Exception as e:
        err_msg = f"    ⚠ Error parsing JSON response: {e}"
        if progress:
            progress.print_notice(err_msg)
        else:
            sys.stderr.write(f"{err_msg}\n")
        return {"items": [], "_model_used": model}
