"""Sliding narrative buffer keeping character names and recent dialogue for context continuity."""

from typing import Any, Dict, List, Tuple


class NarrativeMemory:
    """Sliding narrative buffer keeping character names and recent dialogue for context continuity."""
    def __init__(self, max_pages: int = 3):
        self.max_pages = max_pages
        self.history: List[Tuple[int, List[Dict[str, Any]]]] = []

    def add_page(self, page_num: int, items: List[Dict[str, Any]]) -> None:
        self.history.append((page_num, items))
        if len(self.history) > self.max_pages:
            self.history.pop(0)

    def get_context_prompt(self) -> str:
        if not self.history:
            return ""
        lines = ["\nRECENT NARRATIVE CONTEXT (for pronoun, tone, and character name consistency):"]
        for p_num, items in self.history:
            dialogues = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                if it.get("bubble_type") == "sfx":
                    continue
                t = it.get("translated_text", "")
                if isinstance(t, list):
                    t_str = " ".join(str(x) for x in t).strip()
                else:
                    t_str = str(t or "").strip()
                if t_str:
                    dialogues.append(f'"{t_str}"')
            if dialogues:
                lines.append(f"  - Page {p_num}: " + " | ".join(dialogues[:4]))
        if len(lines) <= 1:
            return ""
        return "\n".join(lines)
