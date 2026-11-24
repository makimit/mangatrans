"""Prompt generation for Gemini Vision scanlation."""


def build_translate_prompt(
    src_lang_name: str,
    tgt_lang_name: str,
    src_lang_code: str,
    narrative_context: str = "",
) -> str:
    """
    Dynamically generates high-precision scanlation prompt respecting reading direction,
    speech bubbles, balloon shapes, conjoined lobes, and professional scanlation rules.
    """
    is_rtl = (src_lang_code in ("ja", "jpn", "ar", "ara", "he", "heb"))
    reading_order_str = (
        "Right-to-Left, Top-to-Bottom (standard manga RTL)"
        if is_rtl
        else "Left-to-Right, Top-to-Bottom (standard manhwa/webtoon/comic LTR)"
    )

    prompt = f"""You are an elite, professional manga and comic scanlation director and {src_lang_name}-to-{tgt_lang_name} translator.
Analyze this manga/comic page with extreme precision and produce a clean scanlation translation JSON.

=== STRICT SCANLATION RULES ===
1. CONJOINED, STACKED, OR MULTI-LOBE SPEECH BUBBLES (CRITICAL):
   - When multiple speech bubbles are stacked, attached, touching, or conjoined in a chain (e.g. multi-lobe balloons), detect EACH individual bubble lobe/balloon as its own SEPARATE item with its own precise [ymin, xmin, ymax, xmax] bounding box and translation!
   - Example: If an upper bubble lobe contains text and a connected lower lobe contains another phrase, output Item A for the upper lobe and Item B for the lower lobe.

2. WHAT TO DETECT & TRANSLATE:
   - ALL speech bubbles without exception (oval, round, conjoined lobes, spiky scream, scalloped cloud, wavy/whisper, dark/inverted balloons).
   - ANY text inside a speech bubble (dialogue, exclamations, calls, cheers, interjections like "ほらっ", "どぉーん♪", "！", etc.) MUST ALWAYS be extracted and translated!
   - ALL narration / caption boxes (rectangular boxes with narrator text, scene setting, or time stamps).
   - ALL significant floating dialogue or monologue text written directly on the page without a bubble container.
   - Reading order: {reading_order_str}.

3. WHAT TO PRESERVE AS ARTWORK (DO NOT EXTRACT):
   - Loose artistic sound effects (SFX) drawn directly across character artwork or scenery outside speech bubbles (e.g. onomatopoeia like ドキドキ, ぼふっ, ふぅ, ざわざわ).
   - Small handwritten background expressions or mood particles outside bubbles (e.g. tiny sigh particles, gasps, tiny mood marks like はぁ..., ッ, えっ in the drawing background). Keep these intact as authentic artwork unless they are actual dialogue sentences.
   - NEVER ignore text that is enclosed inside a speech bubble or caption box.

4. ACCURACY & FIDELITY:
   - Complete coverage: Ensure NO speech bubble or floating dialogue sentence is missed.
   - 100% uncensored fidelity: Faithfully translate all adult, intimate, vulgar, slang, or emotional expressions with exact nuance.

5. BOUNDING BOXES (box_2d):
   - [ymin, xmin, ymax, xmax] normalized on a 0-1000 scale.
   - For speech bubbles ('bubble' or 'box'): The bounding box MUST tightly encompass the full inner cavity and border of that specific bubble balloon/lobe.
   - For floating dialogue ('floating'): The bounding box must cover the exact extent of the spoken text line/paragraph.{narrative_context}

Return ONLY a valid JSON object matching this schema:
{{
  "items": [
    {{
      "box_2d": [ymin, xmin, ymax, xmax],
      "original_text": "Source dialogue text",
      "translated_text": "{tgt_lang_name} translation",
      "bubble_type": "dialogue",
      "container_type": "bubble",
      "bg_color": "white"
    }}
  ]
}}
bubble_type: dialogue | thought | scream | whisper | narration | aside | monologue
container_type: bubble | box | floating
bg_color: white | black
Respond ONLY with JSON."""
    return prompt
