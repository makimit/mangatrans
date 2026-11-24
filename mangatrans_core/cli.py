#!/usr/bin/env python3
"""
mangatrans — Next-Generation Multimodal Manga, Manhwa & Comic Scanlation Suite.

Features:
  1. Multimodal AI Scanlation: High-fidelity dialogue detection, translation, and tone preservation.
  2. 360-Degree Bubble Preservation: Strict barrier-constrained floodfill with zero border clipping.
  3. Dynamic Lobe Segmentation: Multi-lobe conjoined bubble detection with minimum readability capacity checks.
  4. Screentone & Texture-Aware Inpainting: Artifact-free erasure for floating and unboxed dialogue.
  5. Typographic Lettering Engine: Syllabic hyphenation, multi-ratio line wrapping, and halo outlines.
  6. Dynamic Multi-Key Pool: Sequential cycling (1 -> 2 -> ... -> N), explicit key selection (-k), and persistent cooldowns.
  7. Resumable Batch Pipeline: Overlapped prefetching, .partial progress recovery, and format conversion.
"""

import argparse
import os
import sys
import textwrap

# Ensure package root is always in sys.path even when called via symlink or relative path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mangatrans_core.config import (
    VERSION,
    _bold,
    _cyan,
    _dim,
    _green,
    _red,
    _yellow,
    download_scanlation_fonts,
    get_prog_name,
    get_terminal_width,
    mask_key,
)
from mangatrans_core.keys import (
    DynamicKeyManager,
    add_or_update_key,
    check_deps,
    list_keys,
    remove_key,
    reset_key_cooldown,
    test_key_health,
    update_key_entry,
)
from mangatrans_core.pipeline import convert_manga_container, process_manga


def build_parser() -> argparse.ArgumentParser:
    prog = get_prog_name()
    _w = min(get_terminal_width(88), 88)
    _sep = "─" * _w

    description = textwrap.dedent(f"""\
        {_sep}
          {prog}  —  Manga, Manhwa & Comic Translation Suite
        {_sep}
        Translates, inpaints, and typesets manga/manhwa with scanlation-grade
        typography, 360° border-preserved bubble cleaning, texture-aware inpainting,
        conjoined bubble handling, dynamic multi-key failover, and automatic resume.
    """)

    epilog = textwrap.dedent(f"""\
        {_sep}
          BASIC TRANSLATION
        {_sep}
          {prog} manga.cbz                         Translate Japanese → English (default)
          {prog} manga.cbz -t es                   Translate to Spanish
          {prog} manhwa.zip -s ko -t en            Korean manhwa → English
          {prog} page.png -o page_en.png           Single image (any format)
          {prog} /raw_folder/ -t fr -f cbz         Image folder → French CBZ
          {prog} manga.cbz --skip-cover            Preserve original cover page

        {_sep}
          SELECTIVE RE-TRANSLATION  (fix specific pages)
        {_sep}
          {prog} manga.cbz --pages 7 --retrans manga.en.cbz
              Re-do only page 7, merge back into existing output.

          {prog} manga.cbz --pages 5-10 --retrans manga.en.cbz
              Re-do pages 5 through 10.

          {prog} manga.cbz --pages -3 --retrans manga.en.cbz
              Re-do pages 1 through 3.

          {prog} manga.cbz --pages 8- --retrans manga.en.cbz
              Re-do page 8 to the end.

          {prog} manga.cbz --pages 2,5,11-15 --retrans manga.en.cbz
              Re-do scattered pages and a range — any combination.

        {_sep}
          RESUME & BATCH
        {_sep}
          {prog} manga.cbz --resume manga.en.partial.cbz   Resume after interruption
          {prog} vol1.cbz vol2.cbz vol3.cbz -t en          Batch translate multiple files
          {prog} --convert manga.cbz -f pdf                 Format convert (no translation)

        {_sep}
          API KEY MANAGEMENT & SELECTION
        {_sep}
          {prog} manga.cbz -k 2                     Use only Key #2 from saved pool
          {prog} manga.cbz -k 2,3,4                 Use Keys #2, #3, #4 sequentially with failover
          {prog} manga.cbz -k 2-5                   Use Keys #2 through #5 sequentially
          {prog} manga.cbz -k "account1,account2"   Use saved keys by their unique labels
          {prog} manga.cbz -k AIzaSy...             Use a raw one-shot API key directly
          {prog} manga.cbz -k 2 -k AIzaSy...        Mix saved keys and raw keys
          {prog} --add-key AIzaSy... --label "account1"
          {prog} --list-keys
          {prog} --test-key all
          {prog} --reset-cooldown
          {prog} --remove-key 2
          {prog} --update-key "account1" --new-label "primary"

        {_sep}
          SETUP
        {_sep}
          {prog} --check-deps                    Verify Python, Pillow, OpenCV, ffmpeg
          {prog} --download-fonts                Install high-quality scanlation fonts
        {_sep}
    """)

    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description,
        epilog=epilog,
    )

    parser.add_argument(
        "inputs", nargs="*",
        help="Manga archive (.cbz/.cbr/.zip/.pdf/.tar), image file, or folder of images.",
    )

    # ── Translation & Typesetting ─────────────────────────────────────────────
    trans = parser.add_argument_group("Translation & Typesetting")
    trans.add_argument(
        "-s", "--source-lang", default="Japanese", metavar="LANG",
        help="Source language — name or code (default: Japanese / ja).",
    )
    trans.add_argument(
        "-t", "--target-lang", default="English", metavar="LANG",
        help="Target language — name or code (default: English / en).",
    )
    trans.add_argument(
        "-f", "--format", metavar="FMT",
        choices=["cbz", "cbr", "rar", "zip", "pdf", "tar", "folder", "webp", "png", "jpg"],
        default=None,
        help="Output format: cbz cbr rar zip pdf tar folder webp png jpg  (default: mirrors input).",
    )
    trans.add_argument(
        "-o", "--output", metavar="PATH", default=None,
        help="Custom output path (single input only).",
    )
    trans.add_argument(
        "--skip-cover", "--keep-cover", action="store_true",
        help="Preserve the original cover page (Page 1) without translating.",
    )
    trans.add_argument(
        "--export-script", metavar="FILE",
        help="Export the full translated dialogue to a Markdown file.",
    )
    trans.add_argument(
        "--export-json", "--save-json", metavar="PATH", default=None,
        help="Export the raw Gemini translation JSON response (file or directory).",
    )
    trans.add_argument(
        "--keep-upscaled", action="store_true", default=False,
        help="Keep the upscaled resolution in the final output instead of downscaling back to original resolution.",
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_grp = parser.add_argument_group("Resume")
    resume_grp.add_argument(
        "--resume", metavar="PARTIAL_FILE", default=None,
        help="Resume from an existing .partial file (created after an interruption).",
    )

    # ── Selective Re-Translation ──────────────────────────────────────────────
    pages_grp = parser.add_argument_group("Selective Re-Translation")
    pages_grp.add_argument(
        "--pages", metavar="SPEC", default=None,
        help=(
            "Re-translate only specific pages and merge the result.\n"
            "Formats (comma-separated): N  N-M  -M (1..M)  N- (N..end)\n"
            "Examples:  7       5-10       -3       8-       2,5,11-15"
        ),
    )
    pages_grp.add_argument(
        "--retrans", metavar="EXISTING_OUTPUT", default=None,
        help=(
            "Existing translated file to use as the base.\n"
            "Non-selected pages are taken from here; selected pages are re-translated.\n"
            "If omitted, non-selected pages are copied from the original input."
        ),
    )

    # ── Standalone Conversion ────────────────────────────────────────────────
    conv = parser.add_argument_group("Standalone Format Conversion")
    conv.add_argument(
        "--convert", action="store_true",
        help="Convert format without translating. Combine with -f to pick target format.",
    )

    # ── Fonts ─────────────────────────────────────────────────────────────────
    fonts = parser.add_argument_group("Font Management")
    fonts.add_argument(
        "--download-fonts", "--setup-fonts", action="store_true",
        help="Download and install the complete high-quality scanlation font suite.",
    )
    fonts.add_argument(
        "--fonts-dir", metavar="DIR", default=None,
        help="Custom font directory (default: ~/.config/mangatrans/fonts).",
    )

    # ── API Keys & Diagnostics ───────────────────────────────────────────────
    diag = parser.add_argument_group("API Keys & Diagnostics")
    diag.add_argument(
        "-k", "--key", "--keys", "--use-key", "--use-keys",
        action="append", dest="key", metavar="KEY_SPEC", default=None,
        help="Specify key(s) to use sequentially: serial # (e.g. 2, 2-5), unique label (e.g. account1), or raw key. Comma-separated or repeated.",
    )
    diag.add_argument(
        "--add-key", metavar="KEY",
        help="Add a Gemini API key to the persistent pool (requires --label).",
    )
    diag.add_argument(
        "--label", metavar="NAME",
        help="Label for --add-key.",
    )
    diag.add_argument(
        "--update-key", metavar="ID_OR_LABEL",
        help="Update a key's value or label (use with --new-key / --new-label).",
    )
    diag.add_argument(
        "--new-key", metavar="KEY",
        help="Replacement API key string for --update-key.",
    )
    diag.add_argument(
        "--new-label", metavar="LABEL",
        help="Replacement label for --update-key.",
    )
    diag.add_argument(
        "--remove-key", metavar="ID_OR_LABEL",
        help="Remove a key by serial number (#), label, or key string.",
    )
    diag.add_argument(
        "--reset-cooldown", nargs="?", const="all", metavar="ID_OR_LABEL",
        help="Reset rate-limit cooldown for a key or all keys (default: all).",
    )
    diag.add_argument(
        "--test-key", nargs="?", const="all", metavar="ID_OR_LABEL",
        help="Test API key connectivity — by #, label, or 'all'.",
    )
    diag.add_argument(
        "--list-keys", action="store_true",
        help="List all configured API keys with status and cooldown info.",
    )
    diag.add_argument(
        "--check-deps", action="store_true",
        help="Check Python, Pillow, OpenCV, ffmpeg, and API connectivity.",
    )
    diag.add_argument("-v", "--version", action="version", version=f"{prog} v{VERSION}")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Key management commands ───────────────────────────────────────────────
    if args.add_key:
        print(add_or_update_key(args.add_key, args.label))
        sys.exit(0)

    if args.update_key:
        sys.exit(0 if update_key_entry(args.update_key, new_key=args.new_key, new_label=args.new_label) else 1)

    if args.remove_key:
        sys.exit(0 if remove_key(args.remove_key) else 1)

    if args.reset_cooldown is not None:
        sys.exit(0 if reset_key_cooldown(args.reset_cooldown) else 1)

    if args.test_key is not None:
        test_key_health(args.test_key)
        sys.exit(0)

    if args.list_keys:
        list_keys()
        sys.exit(0)

    if args.check_deps:
        check_deps()
        sys.exit(0)

    if getattr(args, "download_fonts", False):
        download_scanlation_fonts(custom_fonts_dir=args.fonts_dir)
        sys.exit(0)

    if not args.inputs:
        parser.print_help()
        sys.exit(1)

    first_target = args.inputs[0]

    # ── Standalone format conversion ──────────────────────────────────────────
    if args.convert or first_target.lower() == "convert":
        target = args.inputs[1] if first_target.lower() == "convert" and len(args.inputs) > 1 else first_target
        fmt = args.format or "cbz"
        convert_manga_container(target, output_format=fmt, output_path=args.output)
        sys.exit(0)

    # ── Validate key pool ─────────────────────────────────────────────────────
    key_mgr = DynamicKeyManager(args.key)
    if not key_mgr.pool:
        prog = get_prog_name()
        if args.key:
            raw_spec = ", ".join(args.key) if isinstance(args.key, list) else str(args.key)
            print(_red(f"✘ Error: Specified API key(s) '{raw_spec}' could not be resolved or found."), file=sys.stderr)
        else:
            print(_red("✘ No API keys configured.") + f"  Run:  {prog} --add-key <KEY> --label <NAME>", file=sys.stderr)
        sys.exit(1)

    # ── Warn about -o with multiple inputs ────────────────────────────────────
    if args.output and len(args.inputs) > 1:
        sys.stderr.write(_yellow(
            "  ⚠  -o/--output is ignored when multiple input files are given.\n"
            "     Each file is saved alongside its input with the default name.\n"
        ))

    # ── Warn about --retrans with multiple inputs ─────────────────────────────
    if getattr(args, "retrans", None) and len(args.inputs) > 1:
        sys.stderr.write(_yellow(
            "  ⚠  --retrans with multiple inputs: it will be applied to every file.\n"
        ))

    # ── Translation loop ──────────────────────────────────────────────────────
    for target in args.inputs:
        key_mgr.reset_cycle()
        try:
            process_manga(
                input_path=target,
                output_path=args.output if len(args.inputs) == 1 else None,
                target_lang_input=args.target_lang,
                source_lang_input=args.source_lang,
                output_format=args.format,
                key_manager=key_mgr,
                export_script_path=args.export_script,
                export_json_path=getattr(args, "export_json", None),
                skip_cover=args.skip_cover,
                resume_path=args.resume,
                page_selection=getattr(args, "pages", None),
                retrans_path=getattr(args, "retrans", None),
                keep_upscaled=getattr(args, "keep_upscaled", False),
            )
        except KeyboardInterrupt:
            sys.stderr.write(_yellow("\n\n  ✗ Interrupted.\n\n"))
            sys.exit(130)
        except SystemExit:
            raise
        except Exception as e:
            sys.stderr.write(_red(f"\n  ✘ Error processing '{target}': {e}\n\n"))
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write(_yellow("\n\n  ✗ Interrupted — exiting.\n\n"))
        sys.exit(130)
