"""mangatrans_core - Manga scanlation translation, vision inpainting, and typesetting toolkit."""

from .config import VERSION
from .containers import extract_input_pages, get_partial_filepath, pack_output_pages
from .keys import (
    AllKeysExhaustedError,
    DynamicKeyManager,
    add_or_update_key,
    check_deps,
    collect_all_configured_keys,
    list_keys,
    parse_key_selection,
    remove_key,
    reset_key_cooldown,
    resolve_key_index,
    test_key_health,
    update_key_entry,
)
from .pipeline import (
    convert_manga_container,
    is_actual_caption_box,
    is_sfx_item,
    load_resumed_pages,
    process_manga,
    render_page_pipeline,
)

__version__ = VERSION

__all__ = [
    "VERSION",
    "DynamicKeyManager",
    "AllKeysExhaustedError",
    "collect_all_configured_keys",
    "resolve_key_index",
    "parse_key_selection",
    "add_or_update_key",
    "update_key_entry",
    "remove_key",
    "reset_key_cooldown",
    "test_key_health",
    "list_keys",
    "check_deps",
    "extract_input_pages",
    "pack_output_pages",
    "get_partial_filepath",
    "load_resumed_pages",
    "process_manga",
    "convert_manga_container",
    "render_page_pipeline",
    "is_sfx_item",
    "is_actual_caption_box",
]
