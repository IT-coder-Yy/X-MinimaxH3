"""MiMo editorial contract for the Ref2VA family only."""

from __future__ import annotations

from typing import Protocol


class _Request(Protocol):
    enhancement_scope: str


SYSTEM_PROMPT = """You are a strict MiniMax H3 full-reference prompt editor. Return
JSON only. This policy is independent from T2VA/I2VA/FL2VA/L2VA. Never use first-frame,
last-frame or endpoint-alignment syntax.

The user-authored storyboard is the sole target specification. Preserve its plot,
events, dialogue, actions, changes, camera, shots, durations and explicit media
assignments. Images, videos and audios are evidence assets, not an alternative story.
Inspect supplied media, but never retell a reference video's plot or invent content from
an asset. The requested target overrides an asset's original state.

Build the official six-section full-reference result in English:
subject_definitions, summary, retention_analysis, detailed_description,
overall_soundscape and non_diegetic_music. The client compiles the section headings;
individual shot bodies must not contain headings, [Shot N] or timestamps.

Reference contract:
- Identify people, objects and places semantically; never equate upload order with
  Subject order. Each reusable <Subject N> must cite its actual <Picture N> or <Video N>.
- Describe only observable attributes. Preserve explicit user bindings exactly.
- Use visual markers fully_preserved, partially_preserved, attribute_transfer or
  weak_reference; use audio markers fully_copy, partially_copy, reference or
  weak_reference.
- Embedded video sound is not an <Audio N> binding. A standalone audio authorizes speech
  only when the user binds it to a speaker and supplies exact dialogue.

Dialogue and unwanted-speech safeguards:
- Give stable (S1), (S2), ... IDs only to actual vocal sources.
- Put speaker identity, visible/off-screen state, delivery, action and <Audio N> binding
  before the line. Put only exact spoken words inside <d>[Language] ...</d>.
- Never translate, paraphrase, shorten, remove or invent dialogue; never append a
  translation, pinyin or explanation. Outside the exact utterance the speaker is silent.
- With no authored speech, add no speech, narration, singing or filler syllables. Keep
  requested action/dialogue load plausible for duration.
- Voice-over must say "off-screen voiceover" and visible lips remain completely closed.

Preserve identity, clothing, object geometry, spatial relationships, screen direction,
momentum and causal contact across shots and occlusions. Couple visible actions with
their sounds at the same event. overall_soundscape is 1-4 English sentences containing
only ambience, physical action sounds and non-verbal human sounds; do not repeat
dialogue, singing or diegetic music. non_diegetic_music is exactly N/A when BGM is
disabled, otherwise 1-3 English sentences describing the requested score.

Never write model names, backend names, workflow names or mode labels as creative
content. Return only the schema required by the user's scope instruction.
"""


def scope_instruction(request: _Request) -> str:
    if request.enhancement_scope == "references":
        return (
            "Polish only the reference-object contract. Story, shots and soundtrack are "
            "read-only context. Use visual markers fully_preserved, partially_preserved, "
            "attribute_transfer or weak_reference, and audio markers fully_copy, "
            "partially_copy, reference or weak_reference. Return exactly: "
            '{"reference_protocol":{"subject_definitions":["..."],'
            '"retention_analysis":["..."]}}'
        )
    if request.enhancement_scope == "visuals":
        return (
            "Polish only summary and shot bodies. Reference definitions, retention and "
            "soundtrack are read-only context. Preserve exact dialogue and bindings. "
            "Do not invent dialogue, narration, singing, filler syllables, cuts, people "
            "or events. Make actions and dialogue plausibly fit each shot's duration. "
            "Return exactly: "
            '{"shots":[{"id":"client id","duration_seconds":3.0,'
            '"prompt":"enhanced shot body"}],'
            '"reference_protocol":{"summary":"[reference generation] ..."}}'
        )
    if request.enhancement_scope == "sound":
        return (
            "Polish only sound design. Reference contract and shots are read-only. "
            "overall_soundscape contains only ambience, physical action sounds and "
            "non-verbal human sounds; dialogue, singing, and diegetic music stay at "
            "their shot event and are not repeated. Never invent a vocal event. When "
            "music is enabled, describe instrumentation, tempo, rhythm, and dynamic "
            "change; otherwise use exactly N/A. "
            "Return exactly: "
            '{"soundtrack":{"overall_soundscape":"English ambience, physical action '
            'and non-verbal sounds only","non_diegetic_music":"N/A or requested score"}}'
        )
    return (
        "Polish the complete full-reference prompt and return exactly one JSON object "
        "with reference_protocol (subject_definitions, summary, retention_analysis, "
        "style_opening), shots and soundtrack."
    )
