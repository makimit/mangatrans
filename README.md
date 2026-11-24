<p align="center">
  <img src="mangatrans.png" alt="MangaTrans" width="160"/>
</p>

# MangaTrans 🎌

> **Lightweight, Cloud-Powered Manga & Comic Scanlation Engine**  
> *Translates, Inpaints, and Typesets manga with inside-out border defense and dynamic typography — without requiring heavy local GPU models.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](#cross-platform-support)
[![GPU Required](https://img.shields.io/badge/GPU-Zero%20(CPU%20Friendly)-brightgreen.svg)](#why-mangatrans)

---

## Why MangaTrans?

### 1. The Core Motivation
Every manga enthusiast knows the feeling: you stumble upon an incredible manga, but no translated version exists anywhere online. Waiting for official localizations or volunteer scanlation groups can take weeks, months, or even years — and many niche titles are never picked up at all.

**MangaTrans was built to bridge that gap.** It is an automated tool created to give you a clean, readable translation quickly, so you can fulfill the desire to read and enjoy the story immediately rather than waiting indefinitely.

### 2. Realistic Expectations & Honest Disclaimer
> [!IMPORTANT]
> **MangaTrans will not produce 100% flawless results on every single page.**
> 
> Manga is an extraordinarily complex, hand-drawn artistic medium featuring stylized calligraphy, decorative announcement balloons, transparent screentones, and vertical right-to-left dialogue. 
> 
> There **will** be times when the tool underperforms: an OCR pass might misinterpret an artistic cursive character, an inpainting mask might leave faint screentone remnants on textured backgrounds, or a line wrap might not match a human typesetter's artistic touch. 
> 
> MangaTrans is engineered as an **automated reading companion for personal enjoyment**, not an absolute replacement for dedicated human scanlation groups who spend hours hand-cleaning and lettering each page.

### 3. Lightweight Cloud Intelligence vs. Heavy Local Models
Most existing open-source manga translation projects require downloading **15 to 30 GB of heavy local PyTorch model weights** (Manga-OCR, ComicTextDetector, LaMa Inpainter, Real-ESRGAN). Running them demands a dedicated NVIDIA GPU with 8–16+ GB of VRAM, complex CUDA driver configurations, and high electricity/compute usage.

MangaTrans takes a fundamentally different, lightweight approach:
- **Cloud-Powered Multimodal Semantic Reasoning**: It offloads text detection, visual context comprehension, reading order, and linguistic translation to the **Google Gemini API** (which offers a generous free tier and rapid responses).
- **Zero Local GPU Required**: The local machine only runs lightweight, mathematically precise OpenCV computer vision algorithms and digital typography layout engines.
- **Fast & Battery-Friendly**: Runs effortlessly on everyday laptops, desktop PCs, and CPU-only servers. Processing a page takes only **~1 to 2.5 seconds**.

---

## Key Capabilities

- **360° Inside-Out Speech Bubble Defense**: Enforces a strict $\text{Cavity} \cap \text{Border} = \emptyset$ mathematical invariant. Never cuts, erodes, or clips bubble outlines or neighboring character linework.
- **Dedicated Radial Burst / Announce Balloon Cleaner**: Uses polar aspect-ratio normalized ray-tracing with Fourier harmonics to preserve spiky perimeter tips and internal illustration details.
- **Dedicated Rectangular Caption Box Frame Scanner**: 4-way directional line scanning preserves 1-pixel frame borders with 100% precision.
- **Knuth-Plass Dynamic Programming Typesetting**: Solves optimal line breaking across varying target shapes (elliptical diamond chords for ovals, uniform widths for boxes) to eliminate ugly dangling single words and awkward ladders.
- **Sliding-Window Narrative Memory**: Keeps track of character names, gender pronouns, and story context across consecutive pages to prevent character misidentification.
- **Panel Line & Divider Continuity**: Reconstructs severed vertical and horizontal panel border strokes using Bézier spline interpolation.
- **Dynamic Multi-Key Pool**: Round-robin cycling across multiple Gemini API keys with automatic exponential cooldowns on HTTP 429 rate limits.
- **Resumable Pipeline & Overlapped Prefetching**: Background prefetching of page $N+1$ while processing page $N$. Seamlessly resume interrupted jobs from `.partial.cbz` files.
- **Selective Page Retranslation**: Retranslate specific pages (e.g. `--pages 5,8-10`) and merge them cleanly back into your existing translated archive.

---

## Cross-Platform Support

MangaTrans is built with standard Python and runs natively across all major operating systems:

| Operating System | Terminal / Shell | Execution Methods |
| :--- | :--- | :--- |
| **Linux** (Ubuntu, Arch, Fedora, Debian) | Bash, Zsh | `./mangatrans` or `python3 -m mangatrans_core` |
| **macOS** (Apple Silicon M1/M2/M3 & Intel) | Zsh, Terminal | `./mangatrans` or `python3 -m mangatrans_core` |
| **Windows** (Windows 10 / 11) | PowerShell, Command Prompt | `python -m mangatrans_core` or `mangatrans` |

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Sayan-Das411/mangatrans.git
cd mangatrans
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```
*(Or install in editable mode: `pip install -e .`)*

### 3. Optional System Tools (for Archive Extraction)
Standard `.cbz` and `.zip` archives work out of the box with zero external dependencies. For additional formats:
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt install poppler-utils p7zip-full unrar
  ```
- **macOS (Homebrew)**:
  ```bash
  brew install poppler p7zip unrar
  ```
- **Windows (Chocolatey / Scoop)**:
  ```powershell
  choco install poppler 7zip
  ```

### 4. Install Free Scanlation Fonts
Download standard comic scanlation fonts (`AnimeAce2_Bold`, `ComicNeue`, `NotoSans`) with a single command:
```bash
python3 -m mangatrans_core --download-fonts
```
*(Fonts are safely stored in your user configuration directory `~/.config/mangatrans/fonts/`)*

---

## Configuration & Gemini API Setup

MangaTrans uses the Google Gemini API for cloud vision and translation. You can obtain a free API key in seconds from [Google AI Studio](https://aistudio.google.com/):

### 1. Add API Keys to Your Pool
```bash
# Add a key with a descriptive label
python3 -m mangatrans_core --add-key AIzaSyYourKeyHere... --label "primary"

# You can add multiple keys for automatic failover and higher throughput:
python3 -m mangatrans_core --add-key AIzaSySecondKeyHere... --label "backup"
```

### 2. Verify Your Configuration
```bash
# Test API key health and network connectivity
python3 -m mangatrans_core --test-key all

# Check system dependencies
python3 -m mangatrans_core --check-deps
```

---

## Usage Guide

### Basic Translation
```bash
# Translate a CBZ archive (Japanese to English by default)
python3 -m mangatrans_core manga.cbz

# Translate to another language (e.g. Spanish, French, German)
python3 -m mangatrans_core manga.cbz -t es
python3 -m mangatrans_core manga.cbz -t fr

# Korean Manhwa to English
python3 -m mangatrans_core manhwa.zip -s ko -t en

# Translate a folder of raw page images into a single CBZ
python3 -m mangatrans_core /path/to/raw_pages/ -f cbz -o translated_manga.cbz

# Single image translation
python3 -m mangatrans_core page.jpg -o page_translated.png

# Preserve original cover page without modifying art
python3 -m mangatrans_core manga.cbz --skip-cover
```

### Resuming & Selective Page Retranslation
```bash
# Resume an interrupted translation
python3 -m mangatrans_core manga.cbz --resume manga.en.partial.cbz

# Retranslate only page 7 and merge back into the completed archive
python3 -m mangatrans_core manga.cbz --pages 7 --retrans manga.en.cbz

# Retranslate a page range (pages 5 through 12)
python3 -m mangatrans_core manga.cbz --pages 5-12 --retrans manga.en.cbz
```

### Key Selection on the Fly
```bash
# Use a specific saved key by index or label
python3 -m mangatrans_core manga.cbz -k 2
python3 -m mangatrans_core manga.cbz -k "primary"

# Use a one-shot raw API key directly
python3 -m mangatrans_core manga.cbz -k AIzaSyOneShotKey...
```

---

## Supported Formats

| Format | Extension | Input Support | Output Support |
| :--- | :--- | :---: | :---: |
| **Comic Book ZIP** | `.cbz` | ✔ | ✔ (Default) |
| **Standard ZIP** | `.zip` | ✔ | ✔ |
| **PDF Document** | `.pdf` | ✔ | ✔ |
| **Comic Book RAR** | `.cbr`, `.rar` | ✔ | ✔ (requires `rar`) |
| **Image Folder** | Directory | ✔ | ✔ |
| **Standalone Images** | `.webp`, `.png`, `.jpg` | ✔ | ✔ |

---

## Architecture

For a comprehensive technical deep-dive into the inside-out inpainting mathematics, single-pass cavity extraction, Knuth-Plass dynamic programming formulation, and narrative sliding-window memory, refer to [ARCHITECTURE.md](ARCHITECTURE.md).

---

## License & Fair Use

This project is licensed under the **MIT License**.

> [!NOTE]
> MangaTrans is intended strictly for personal reading, accessibility, and educational purposes. Please support official manga artists, publishers, and localization releases whenever they are available in your region.
