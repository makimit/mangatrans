import base64
import datetime
import io
import json
import os
import re
import shutil
import struct
import sys
import threading
import time
import urllib.request
import unicodedata
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union

VERSION = "1.0.0"

# ------------------------------------------------------------------------------
# 1. PATHS & CONFIGURATION CONSTANTS
# ------------------------------------------------------------------------------
CONFIG_DIR = os.path.expanduser("~/.config/mangatrans")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
FONTS_DIR = os.path.join(CONFIG_DIR, "fonts")

SHARED_KEYS_DIR       = os.path.expanduser("~/.config/gemini-shared")
SHARED_KEYS_FILE      = os.path.join(SHARED_KEYS_DIR, "config.json")
SHARED_COOLDOWNS_FILE = os.path.join(SHARED_KEYS_DIR, "cooldowns.json")

CACHE_DIR = os.path.expanduser("~/.cache/mangatrans")

SUPPORTED_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".avif", ".jxl"
}
SUPPORTED_ARCHIVE_EXTS = {
    ".cbz", ".cbr", ".cb7", ".cbt", ".zip", ".rar", ".7z", ".tar", ".gz"
}
SUPPORTED_DOC_EXTS = {".pdf", ".epub"}

DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

# ------------------------------------------------------------------------------
# 2. LANGUAGE DATABASE & NORMALIZATION
# ------------------------------------------------------------------------------
LANGUAGE_DATABASE = [
    ("English", "en", "eng", ["en", "eng", "english"]),
    ("Spanish", "es", "spa", ["es", "spa", "spanish", "espanol", "español"]),
    ("French", "fr", "fre", ["fr", "fre", "fra", "french", "francais", "français"]),
    ("German", "de", "ger", ["de", "ger", "deu", "german", "deutsch"]),
    ("Japanese", "ja", "jpn", ["ja", "jp", "jpn", "japanese", "nihongo"]),
    ("Chinese (Simplified)", "zh-CN", "chi", ["zh", "zh-cn", "chi", "zho", "chinese", "mandarin", "simplified chinese"]),
    ("Chinese (Traditional)", "zh-TW", "chi", ["zh-tw", "traditional chinese", "taiwanese"]),
    ("Korean", "ko", "kor", ["ko", "kor", "korean", "hangul"]),
    ("Hindi", "hi", "hin", ["hi", "hin", "hindi"]),
    ("Bengali", "bn", "ben", ["bn", "ben", "bengali", "bangla"]),
    ("Arabic", "ar", "ara", ["ar", "ara", "arabic"]),
    ("Portuguese", "pt", "por", ["pt", "por", "portuguese", "portugues", "português"]),
    ("Russian", "ru", "rus", ["ru", "rus", "russian"]),
    ("Italian", "it", "ita", ["it", "ita", "italian", "italiano"]),
    ("Dutch", "nl", "dut", ["nl", "dut", "nld", "dutch", "nederlands"]),
    ("Turkish", "tr", "tur", ["tr", "tur", "turkish", "turkce", "türkçe"]),
    ("Vietnamese", "vi", "vie", ["vi", "vie", "vietnamese", "tieng viet"]),
    ("Thai", "th", "tha", ["th", "tha", "thai"]),
    ("Indonesian", "id", "ind", ["id", "ind", "indonesian", "bahasa indonesia"]),
    ("Polish", "pl", "pol", ["pl", "pol", "polish", "polski"]),
    ("Ukrainian", "uk", "ukr", ["uk", "ukr", "ukrainian"]),
    ("Swedish", "sv", "swe", ["sv", "swe", "swedish", "svenska"]),
    ("Tagalog", "tl", "tgl", ["tl", "tgl", "tagalog", "filipino"]),
]


def resolve_language_info(lang_input: str) -> Tuple[str, str, str]:
    """Normalizes user-supplied language string into (Full Name, Code-2, Code-3)."""
    if not lang_input:
        return ("English", "en", "eng")
    cleaned = lang_input.strip().lower()
    if cleaned in ("same", "original", "source", "verbatim"):
        return ("Original Language", "orig", "und")
    if cleaned == "auto":
        return ("Auto-detected", "auto", "und")
    for name, code1, code2, aliases in LANGUAGE_DATABASE:
        if cleaned == name.lower() or cleaned == code1.lower() or cleaned == code2.lower() or cleaned in aliases:
            return (name, code1.lower(), code2.lower())
    return (lang_input.strip().capitalize(), re.sub(r"[^a-zA-Z0-9]", "", cleaned)[:4] or "sub", "und")


# ------------------------------------------------------------------------------
# 3. ANSI COLOR & TERMINAL HELPERS
# ------------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def _bold(t: str) -> str:    return _c("1", t)
def _dim(t: str) -> str:     return _c("2", t)
def _green(t: str) -> str:   return _c("32", t)
def _yellow(t: str) -> str:  return _c("33", t)
def _cyan(t: str) -> str:    return _c("36", t)
def _red(t: str) -> str:     return _c("31", t)
def _magenta(t: str) -> str: return _c("35", t)
def _blue(t: str) -> str:    return _c("34", t)

ANSI_REGEX = re.compile(r'\x1b\[[0-9;]*m')

def strip_ansi(s: str) -> str:
    return ANSI_REGEX.sub('', s)

def visible_width(s: str) -> int:
    """Calculates true monospace terminal display width, handling double-width Unicode & emojis."""
    s_clean = strip_ansi(s)
    w = 0
    for ch in s_clean:
        status = unicodedata.east_asian_width(ch)
        if status in ('W', 'F'):
            w += 2
        else:
            w += 1
    return w

def pad_visible(s: str, target_width: int, align: str = "left") -> str:
    v_len = visible_width(s)
    pad = max(0, target_width - v_len)
    if align == "right":
        return (" " * pad) + s
    return s + (" " * pad)

def get_prog_name() -> str:
    return os.environ.get("PROG_NAME") or os.path.basename(sys.argv[0]) or "mangatrans"

def mask_key(k: str) -> str:
    if not k:
        return "(none)"
    if len(k) > 12:
        return f"{k[:6]}...{k[-4:]}"
    return k[:3] + "..."

def get_terminal_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default

def print_banner(title: str) -> None:
    width = min(get_terminal_width(), 88)
    sep = _dim("─" * width)
    print(sep)
    print(f"  {_bold(_cyan(get_prog_name()))}  {_dim('—')}  {_bold(title)}")
    print(sep)

def natural_sort_key(s: str) -> List[Union[int, str]]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

def parse_page_selection(spec: str, total: int) -> List[int]:
    selected: Set[int] = set()
    for raw_token in re.split(r"[,\s]+", spec.strip()):
        token = raw_token.strip()
        if not token:
            continue
        m = re.match(r"^(\d+)-(\d+)$", token)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = min(a, b), max(a, b)
            if lo < 1 or hi > total:
                sys.stderr.write(_yellow(f"  ⚠  Page range {token} partially out of bounds (total: {total}), clamped.\n"))
            selected.update(range(max(1, lo), min(total, hi) + 1))
            continue
        m = re.match(r"^-(\d+)$", token)
        if m:
            b = int(m.group(1))
            selected.update(range(1, min(total, b) + 1))
            continue
        m = re.match(r"^(\d+)-$", token)
        if m:
            a = int(m.group(1))
            if a > total:
                sys.stderr.write(_yellow(f"  ⚠  Page {a} is beyond total ({total}), skipped.\n"))
            selected.update(range(max(1, a), total + 1))
            continue
        if re.match(r"^\d+$", token):
            n = int(token)
            if 1 <= n <= total:
                selected.add(n)
            else:
                sys.stderr.write(_yellow(f"  ⚠  Page {n} is out of range (total: {total}), skipped.\n"))
            continue
        sys.stderr.write(_yellow(f"  ⚠  Unrecognised page token: '{token}', skipped.\n"))
    return sorted(selected)

def detect_image_mime_type(image_bytes: bytes) -> str:
    if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(image_bytes) >= 2 and image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if len(image_bytes) >= 6 and image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(image_bytes) >= 2 and image_bytes[:2] == b"BM":
        return "image/bmp"
    if len(image_bytes) >= 4 and image_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return "image/webp"

def format_duration(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"

def format_datetime(iso_str: str) -> str:
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


# ------------------------------------------------------------------------------
# 4. LIVE PROGRESS SPINNER & TIMER
# ------------------------------------------------------------------------------
class LiveProgress:
    """Threaded real-time terminal spinner and elapsed timer with safe logging."""
    def __init__(self, message: str, base_label: Optional[str] = None):
        self.message = message
        self.base_label = base_label or message
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.start_time = time.time()
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def set_message(self, new_message: str) -> None:
        with self.lock:
            self.message = new_message

    def print_notice(self, text: str) -> None:
        with self.lock:
            sys.stdout.write(f"\r\033[K{text}\n")
            sys.stdout.flush()

    def _animate(self) -> None:
        idx = 0
        while not self.stop_event.is_set():
            elapsed = time.time() - self.start_time
            frame = self.frames[idx % len(self.frames)]
            with self.lock:
                term_w = get_terminal_width(80)
                colored_frame = _cyan(frame)
                elapsed_str = _dim(f"({elapsed:.1f}s)")
                line = f"    {self.message} {colored_frame} {elapsed_str}"
                plain_line = f"    {self.message} {frame} ({elapsed:.1f}s)"
                if len(plain_line) > term_w:
                    trim = term_w - len(f" {frame} ({elapsed:.1f}s)") - 4 - 3
                    line = f"    {self.message[:max(0,trim)]}... {colored_frame} {elapsed_str}"
                sys.stdout.write(f"\r\033[K{line}")
                sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

    def start(self) -> "LiveProgress":
        self.thread.start()
        return self

    def stop(self, final_note: str = "done") -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        elapsed = time.time() - self.start_time
        with self.lock:
            is_ok = final_note.startswith("done") or final_note.startswith("preserv")
            is_warn = "exhausted" in final_note or "error" in final_note
            if is_ok:
                note_str = _green(f"✔ {final_note}")
            elif is_warn:
                note_str = _yellow(f"⚠ {final_note}")
            else:
                note_str = final_note
            sys.stdout.write(f"\r\033[K    {self.base_label} {note_str} {_dim(f'({elapsed:.1f}s)')}\n")
            sys.stdout.flush()


# ------------------------------------------------------------------------------
# 5. EMBEDDED FONTS & FONT MANAGEMENT
# ------------------------------------------------------------------------------
EMBEDDED_ANIMEACE_WOFF_B64 = "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"

def unpack_woff_to_ttf(woff_bytes: bytes) -> bytes:
    if len(woff_bytes) < 44 or woff_bytes[:4] != b"wOFF":
        return woff_bytes
    header = woff_bytes[:44]
    sig, flavor, length, num_tables, reserved, total_sfnt_size, major, minor, meta_offset, meta_len, meta_orig_len, priv_offset, priv_len = struct.unpack(">4sIIHHIIHHIIII", header)
    tables = []
    pos = 44
    for _ in range(num_tables):
        tag, offset, comp_len, orig_len, orig_chksum = struct.unpack(">4sIIII", woff_bytes[pos:pos+20])
        tables.append({"tag": tag, "offset": offset, "comp_len": comp_len, "orig_len": orig_len, "orig_chksum": orig_chksum})
        pos += 20
    ttf_out = bytearray()
    ttf_out.extend(struct.pack(">IHHHH", flavor, num_tables, 16, 4, 0))
    current_offset = 12 + num_tables * 16
    table_data_list = []
    for t in tables:
        c_data = woff_bytes[t["offset"]:t["offset"]+t["comp_len"]]
        decomp = zlib.decompress(c_data) if t["comp_len"] != t["orig_len"] else c_data
        padded = decomp + b"\x00" * ((4 - (len(decomp) % 4)) % 4)
        table_data_list.append((t["tag"], t["orig_chksum"], current_offset, t["orig_len"], padded))
        current_offset += len(padded)
    for tag, chksum, offset, length, _ in table_data_list:
        ttf_out.extend(struct.pack(">4sIII", tag, chksum, offset, length))
    for _, _, _, _, data in table_data_list:
        ttf_out.extend(data)
    return bytes(ttf_out)

def ensure_scanlation_fonts(custom_fonts_dir: Optional[str] = None) -> str:
    fonts_dir = custom_fonts_dir or FONTS_DIR
    os.makedirs(fonts_dir, exist_ok=True)
    target_ttf = os.path.join(fonts_dir, "AnimeAce2_Bold.ttf")
    if os.path.isfile(target_ttf) and os.path.getsize(target_ttf) > 1000:
        return target_ttf
    try:
        raw_woff = base64.b64decode(EMBEDDED_ANIMEACE_WOFF_B64)
        ttf_bytes = unpack_woff_to_ttf(raw_woff)
        if len(ttf_bytes) > 1000:
            with open(target_ttf, "wb") as f:
                f.write(ttf_bytes)
            return target_ttf
    except Exception as e:
        sys.stderr.write(f"    ⚠ Warning: Failed to unpack embedded font: {e}\n")
    for f in ["ComicNeue-Bold.ttf", "ComicNeueAngular-Bold.ttf", "anime_ace.ttf"]:
        p = os.path.join(fonts_dir, f)
        if os.path.isfile(p):
            return p
    return target_ttf

def find_system_fallback_font(is_symbol: bool = False) -> str:
    """Finds a reliable cross-platform fallback font across Linux, macOS, and Windows."""
    candidates = []
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        fonts = os.path.join(windir, "Fonts")
        if is_symbol:
            candidates.extend([
                os.path.join(fonts, "seguisym.ttf"),
                os.path.join(fonts, "seguiemj.ttf"),
                os.path.join(fonts, "arialbd.ttf"),
                os.path.join(fonts, "arial.ttf"),
            ])
        else:
            candidates.extend([
                os.path.join(fonts, "arialbd.ttf"),
                os.path.join(fonts, "segoeuib.ttf"),
                os.path.join(fonts, "arial.ttf"),
                os.path.join(fonts, "tahoma.ttf"),
            ])
    elif sys.platform == "darwin":  # macOS
        if is_symbol:
            candidates.extend([
                "/System/Library/Fonts/Apple Symbols.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ])
        else:
            candidates.extend([
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
            ])
    else:  # Linux / BSD
        if is_symbol:
            candidates.extend([
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/local/share/fonts/DejaVuSans-Bold.ttf",
            ])
        else:
            candidates.extend([
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
                "/usr/local/share/fonts/DejaVuSans-Bold.ttf",
            ])
    for c in candidates:
        if os.path.isfile(c) and os.path.getsize(c) > 1000:
            return c
    return ""


def get_font_paths(lang_code: str = "en") -> Tuple[str, str]:
    ensure_scanlation_fonts()
    primary_font = os.path.join(FONTS_DIR, "AnimeAce2_Bold.ttf")
    if lang_code in ("hi", "hin", "mr", "ne", "sa"):
        dev_font = os.path.join(FONTS_DIR, "NotoSansDevanagari.ttf")
        if os.path.isfile(dev_font):
            primary_font = dev_font
    elif lang_code in ("bn", "ben", "as"):
        ben_font = os.path.join(FONTS_DIR, "NotoSansBengali.ttf")
        if os.path.isfile(ben_font):
            primary_font = ben_font
    elif lang_code in ("ru", "rus", "uk", "ukr", "el", "grc", "zh", "ko", "ja"):
        noto_font = os.path.join(FONTS_DIR, "NotoSans.ttf")
        if os.path.isfile(noto_font):
            primary_font = noto_font

    if not os.path.isfile(primary_font) or os.path.getsize(primary_font) < 1000:
        fallback = find_system_fallback_font(is_symbol=False)
        if fallback:
            primary_font = fallback

    sym_font = find_system_fallback_font(is_symbol=True)
    if not sym_font:
        sym_font = primary_font
    return primary_font, sym_font

FONT_PACKAGES = [
    {
        "id": 1,
        "name": "Anime Ace 2 (Bold)",
        "desc": "Classic Comic Lettering (English / Latin)",
        "files": [("AnimeAce2_Bold.ttf", "embedded", "Embedded WOFF v1 Binary")],
    },
    {
        "id": 2,
        "name": "Comic Neue (Bold & Regular)",
        "desc": "Modern Clean Comic Lettering (English / Latin / EU)",
        "files": [
            ("ComicNeue-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Bold.ttf", "Google Fonts (Comic Neue Bold)"),
            ("ComicNeue-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Regular.ttf", "Google Fonts (Comic Neue Regular)"),
        ],
    },
    {
        "id": 3,
        "name": "Noto Sans (Universal)",
        "desc": "Full Multilingual / Latin / Cyrillic / Greek",
        "files": [
            ("NotoSans.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf", "Google Fonts (Noto Sans Variable)"),
        ],
    },
    {
        "id": 4,
        "name": "Noto Sans Devanagari",
        "desc": "Hindi / Marathi / Sanskrit / Nepali",
        "files": [
            ("NotoSansDevanagari.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf", "Google Fonts (Devanagari/Hindi)"),
        ],
    },
    {
        "id": 5,
        "name": "Noto Sans Bengali",
        "desc": "Bengali / Bangla / Assamese",
        "files": [
            ("NotoSansBengali.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansbengali/NotoSansBengali%5Bwdth%2Cwght%5D.ttf", "Google Fonts (Bengali/Bangla)"),
        ],
    },
]

def download_scanlation_fonts(custom_fonts_dir: Optional[str] = None, auto_yes: bool = False) -> None:
    fonts_dir = custom_fonts_dir or FONTS_DIR
    os.makedirs(fonts_dir, exist_ok=True)
    print_banner("Scanlation Fonts Downloader & Setup")
    print(f"  Target Font Directory: {fonts_dir}\n")
    for pkg in FONT_PACKAGES:
        for fname, url, desc in pkg["files"]:
            dest_path = os.path.join(fonts_dir, fname)
            if url == "embedded":
                ensure_scanlation_fonts(fonts_dir)
            elif not os.path.isfile(dest_path) or os.path.getsize(dest_path) < 1000:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        data = resp.read()
                        if len(data) > 500:
                            with open(dest_path, "wb") as out_f:
                                out_f.write(data)
                except Exception as e:
                    sys.stderr.write(f"  ⚠ Failed to download {fname}: {e}\n")
    print("  ✔ Font setup complete.\n")
