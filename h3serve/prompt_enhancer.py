from __future__ import annotations

import base64
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import uuid
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from .contract import ContractError
from .prompt_policies.fl2va import (
    SYSTEM_PROMPT as FL2VA_SYSTEM_PROMPT,
    scope_instruction as fl2va_scope_instruction,
)
from .prompt_policies.ref2va import (
    SYSTEM_PROMPT as REF2VA_POLICY_PROMPT,
    scope_instruction as ref2va_scope_instruction,
)


MIMO_ENDPOINT = os.environ.get(
    "H3_MIMO_ENDPOINT",
    "https://api.xiaomimimo.com/v1/chat/completions",
).strip()
MIMO_MODEL = os.environ.get("H3_MIMO_MODEL", "mimo-v2.5").strip()
MAX_SHOTS = 20
MAX_SHOT_TEXT = 6_000
CONDITION_MODES = {"T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"}
ENHANCEMENT_SCOPES = {"full", "references", "visuals", "sound"}
TOP_LEVEL_H3_FIELDS = (
    "integrated_multimodal_description:",
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
REFERENCE_LABEL_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>", re.IGNORECASE)
SUBJECT_DEFINITION_RE = re.compile(
    r"^<Subject\s+(\d+)>\s*(?::|\bis\b)", re.IGNORECASE
)
AUDIO_DEFINITION_RE = re.compile(
    r"^<Audio\s+(\d+)>\s*(?::|\bis\b)", re.IGNORECASE
)
RETENTION_MARKERS = {
    "fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference",
    "fully_copy", "partially_copy", "reference", "weak_reference",
}
INTERNAL_MODE_PREFIX_RE = re.compile(
    r"^\s*(?:[\[【(（]\s*)?"
    r"(?:MiniMax\s*H3\s*)?"
    r"(?:Ref2VA|FL2VA|L2VA|I2VA|T2VA)"
    r"\s*(?:模式|mode)?\s*(?:[\]】)）]\s*)?"
    r"(?:[，,:：;；\-—]\s*)?",
    re.IGNORECASE,
)

# Kept temporarily as inert migration history for older downstream imports. Runtime
# selection below uses prompt_policies/fl2va.py and prompt_policies/ref2va.py only.
_LEGACY_BASE_SYSTEM_PROMPT_DO_NOT_USE = """You are a strict prompt editor for MiniMax H3 audio-video
generation. Return JSON only. Preserve the completed storyboard's intent, people,
events, exact dialogue, reference bindings, shot count, shot order, durations and BGM
choice. Improve only clarity, continuity and production detail. Do not invent new plot,
people, dialogue, cuts or story events.

Rewrite each shot into concise English production language containing, when present:
visual subject/action/environment, exact speaker-tagged dialogue, camera movement, and
diegetic sound at the moment it occurs. Preserve the source language only inside
<d>[Language] ...</d> dialogue tags, lyrics, or text visibly present in the scene. Keep
character identity, clothing, geography, lighting, and motion continuous across shots
and supplied first/last frames.

H3 reliability rules:
- One shot means one continuous take; do not add an unrequested cut.
- Give only actual vocal sources a stable global ID such as (S1), (S2), reused across
  all shots. Never attach a speaker ID to a character who does not speak, sing, or
  produce an off-screen human voice. Every spoken line names its speaker, on-screen or
  off-screen state, and visible speaking action. Put only the
  exact spoken words inside <d>[Chinese] ...</d> (or the user's actual language); put
  identity, delivery and action outside <d>. Never translate or paraphrase dialogue.
- Never invent dialogue, narration, singing, filler syllables, or an extra vocal event.
  If the user authored no speech, describe the intended audible track positively using
  ambience, physical action sounds and non-verbal human sounds. If the user authored
  speech, the speaker vocalizes only for the natural duration of the exact <d> text;
  outside that utterance the speaker remains silent. Keep the requested action and
  dialogue load plausible for the allocated shot duration rather than filling unused
  time with speech.
- Voice-over uses the words "off-screen voiceover" and explicitly says the visible
  character's lips remain completely closed. If speech crosses a cut, mark continuity
  with <scenetrans>; use <cutoff> only when the requested line truly ends at the video.
- Couple action and its sound in the same shot. Do not replace concrete diegetic sound
  with vague "natural sound".
- If BGM is disabled, soundtrack.non_diegetic_music must be exactly "N/A". Do not add
  any unrequested music, singing, narration or new spoken lines. BGM being disabled
  never means dialogue is disabled: preserve every user-written spoken line exactly.
  Fill soundtrack.overall_soundscape with only scene-grounded ambience, physical action
  sounds, and non-verbal human sounds. Dialogue, singing, and diegetic music belong at
  their exact event in the shot and must not be repeated in overall_soundscape.
- If BGM is enabled, retain the requested music style and keep it below dialogue.
- Respect the supplied H3 mode. T2VA has no image anchor; I2VA develops forward from
  the opening image; FL2VA describes a continuous observable path from opening to
  ending image; L2VA infers a plausible opening and converges to the ending image;
  Ref2VA uses the user's local references as identity/style/object evidence, not as
  first/last-frame anchors. If those references are not attached here, do not invent
  their contents; preserve the user's own reference descriptions.
- Use supplied images only as continuity evidence. Describe only details actually
  observable in an attached image. If a detail cannot be confirmed, preserve the
  user's wording and do not invent identity, clothing, setting, action or composition.
- Return only shot bodies. Do not emit [Shot N], timestamps, Picture alignment lines,
  integrated_multimodal_description, overall_soundscape or non_diegetic_music inside
  a shot prompt; the deterministic H3 compiler owns that syntax.

Dialogue preservation is mandatory. Convert quoted dialogue into H3 dialogue tags;
never summarize it as "speaks", "opens their mouth", "responds" or "the two converse".
For example, preserve these two source lines as:
女孩开心地问：\n<d>[Chinese] 叔叔，这个书包是送给我的吗？</d>
男人微笑着回答：\n<d>[Chinese] 是的，背上试试看。</d>
Do not translate, paraphrase, shorten or remove the words inside the tags.

Sound fields follow the official division. overall_soundscape is one continuous English
paragraph of 1-4 sentences covering only ambience, physical action sounds, and
non-verbal human sounds across the video. It is N/A only for requested complete silence.
non_diegetic_music is N/A when audience-only music is not requested; otherwise it uses
1-3 English sentences describing instrumentation, tempo, rhythm, and dynamic change.
Do not use either sound field to repeat dialogue or to invent a vocal event.

The user message specifies the only JSON block that may be returned for this request.
For a scoped request, return its exact partial schema rather than the other modules.
The following complete shape applies only to a full enhancement request:
{
  "shots": [{"id": "client id", "duration_seconds": 3.0,
             "prompt": "enhanced shot text"}],
  "soundtrack": {"overall_soundscape": "...",
                 "non_diegetic_music": "N/A or requested style"}
}
"""

_LEGACY_REF2VA_SYSTEM_PROMPT_DO_NOT_USE = """You polish a complete short-video storyboard into a precise
full-reference generation prompt. Return JSON only.

The user-authored storyboard is the sole target specification. It decides the plot,
events, dialogue, actions, changes, camera, shot count, shot order, durations and
explicit media assignments. Preserve it. Reference media are source assets, not an
alternative storyboard and not a narrative to recap. Use them only to ground the
identity, object design, location, motion or camera characteristics that the user asks
to retain or transform. Never introduce an event, line of dialogue, relationship or
shot merely because it appears in an image or video reference. When the target
storyboard and a reference differ, write the requested target result rather than the
reference's original state.

Inspect supplied images and videos, and listen to supplied audio before writing. For a
video, use its visible subjects, motion, camera behavior and temporal continuity as
evidence; do not retell its sequence of events. Its embedded soundtrack is not an
audio-reference binding. Improve the target storyboard's clarity, continuity and H3
reliability without changing its intent, exact dialogue or explicit reference
assignment.

Identify which references show people, objects or places instead of treating upload
order as subject order. Make each user-authored shot executable by clarifying the
already requested action, camera behavior and synchronized scene sound. Bind an audio
assigned as a voice reference to the concrete visible speaker specified by the
storyboard. Audio assigned as ambience, effects, reused soundtrack or background-music
reference keeps that role and must not be forced into a speaking event.

The reference-media list is authoritative only for the available labels and media
types. Use concise subject definitions and retention rows to explain how a referenced
asset contributes to the requested target; do not use those sections to caption,
summarize or reconstruct the reference. The effective duration in the user payload is
authoritative; repeat that duration in summary rather than guessing from a default
frame count.

Do not assign subjects by upload order. For example, if <Picture 1> is a bag,
<Picture 2> is a girl and <Picture 3> is a man, sensible IDs may be the girl as
<Subject 1> (S1), the man as <Subject 2> (S2), and the bag as <Subject 3>.
Describe only visible attributes actually present in each image. Listen to each audio
for timbre and delivery, but do not copy or quote words from the reference recording.
An explicit user assignment such as "S1 uses <Audio 2>" always overrides your guess.

For every audio that the user assigns as a voice reference, write both:
1. one definition such as "<Audio 1> is the voice-timbre reference for <Subject 1>
   (S1)"; and
2. one actual speaking instruction inside a shot, naming that same Subject and
   <Audio N>, followed by the new line inside <d>[Language] ...</d>.

Dialogue serialization is a strict compiler contract, not a writing-style choice.
Convert every spoken line from the source storyboard, including lines originally
written in Chinese quotation marks, into exactly this form:
"可见的女孩 <Subject 1> (S1) 使用 <Audio 1> 的音色开心地问：\n"
"<d>[Chinese] 叔叔，这个书包是送给我的吗？</d>"
Only the words that will actually be spoken may appear between <d> and </d>. Keep
those words byte-for-byte equivalent to the user's original line. Never surround the
line with quotation marks. Never append a translation, pinyin, gloss, explanation or
parenthetical text after it. In particular, this is forbidden:
"叔叔，这个书包是送给我的吗？" (Uncle, is this backpack for me?).
Use [Chinese] for Chinese dialogue and the actual language name for other dialogue.

Keep identity, clothing, objects, location and spatial continuity stable between shots.
Each shot is one continuous take. Put speaker identity and speaking action before the
dialogue. Couple visible actions with their sounds. When music is disabled, return
non_diegetic_music as exactly N/A and use only dialogue and scene-grounded sound.

Give (S1), (S2), ... IDs only to actual vocal sources. Never invent dialogue, narration,
singing, filler syllables, or an extra vocal event. A reference audio file does not by
itself authorize speech: it produces dialogue only when the user's shot explicitly binds
it to a visible or off-screen speaker and supplies the exact words. When speech is
requested, the speaker vocalizes only for the natural duration of the exact <d> text and
remains silent outside that utterance. Keep the requested action and dialogue load
plausible for the allocated shot duration.

overall_soundscape contains only ambience, physical action sounds, and non-verbal human
sounds in one continuous English paragraph of 1-4 sentences. Do not repeat dialogue,
singing, or diegetic music there. non_diegetic_music is exactly N/A when audience-only
music is not requested; otherwise describe instrumentation, tempo, rhythm, and dynamic
change in 1-3 English sentences.

The client compiles six protocol sections: subject_definitions, summary,
retention_analysis, detailed_description, overall_soundscape, and
non_diegetic_music. Following MiniMax's official full-reference rewrite format, write
all six sections in English. Preserve the source language only for dialogue or lyrics
inside <d>[Language] ...</d> and for text visibly present in the scene.
Never write MiniMax, H3, Ref2VA, FL2VA, L2VA, I2VA, T2VA, "mode", "模式", backend,
engine or workflow as creative content. Do not put [Shot N], timestamps or section
headings inside an individual shot prompt.

Use sequential <Subject N> IDs chosen by semantic identity. Every Subject definition
must cite its actual <Picture N> or <Video N>. Define a standalone Picture, Video or
Audio when the asset itself is used directly later. Use the official visual retention
markers fully_preserved, partially_preserved, attribute_transfer and weak_reference;
attribute_transfer applies only when attributes move to a different identifiable
target. Use fully_copy, partially_copy, reference or weak_reference for audio. The
user message specifies the only JSON block that may be returned for this request.
"""

# Backwards-compatible name. New code must import the family policy directly.
SYSTEM_PROMPT = FL2VA_SYSTEM_PROMPT


@dataclass(frozen=True)
class EnhancementRequest:
    shots: tuple[dict[str, Any], ...]
    bgm_enabled: bool
    bgm_style: str
    condition_mode: str
    effective_duration_seconds: float
    reference_media: tuple[dict[str, str], ...] = ()
    enhancement_scope: str = "full"
    reference_protocol: dict[str, Any] | None = None
    soundtrack: dict[str, str] | None = None


def parse_enhancement_request(raw: str) -> EnhancementRequest:
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContractError("shots must be valid JSON") from error
    if not isinstance(document, dict):
        raise ContractError("shots must be a JSON object")
    shots = document.get("shots")
    if not isinstance(shots, list) or not 1 <= len(shots) <= MAX_SHOTS:
        raise ContractError(f"shots must contain 1 to {MAX_SHOTS} items")
    normalized: list[dict[str, Any]] = []
    total = 0.0
    for index, item in enumerate(shots, 1):
        if not isinstance(item, dict):
            raise ContractError(f"shot {index} must be an object")
        prompt = str(item.get("prompt", "")).strip()
        if len(prompt) > MAX_SHOT_TEXT:
            raise ContractError(f"shot {index} prompt must be at most {MAX_SHOT_TEXT} characters")
        if not prompt:
            raise ContractError(f"shot {index} must be written before polishing")
        try:
            duration = float(item.get("duration_seconds"))
        except (TypeError, ValueError) as error:
            raise ContractError(f"shot {index} duration must be numeric") from error
        if not 0.5 <= duration <= 15:
            raise ContractError(f"shot {index} duration must be between 0.5 and 15 seconds")
        total += duration
        normalized.append({
            "id": str(item.get("id") or f"shot-{index}"),
            "duration_seconds": round(duration, 2),
            "prompt": prompt,
        })
    if total > 15.001:
        raise ContractError("total shot duration cannot exceed 15 seconds")
    bgm_enabled = bool(document.get("bgm_enabled", False))
    bgm_style = str(document.get("bgm_style", "")).strip()
    if bgm_enabled and not bgm_style:
        raise ContractError("BGM style is required when BGM is enabled")
    condition_mode = str(document.get("condition_mode", "T2VA")).strip().upper()
    if condition_mode not in CONDITION_MODES:
        raise ContractError(
            "condition_mode must be T2VA, I2VA, FL2VA, L2VA or REF2VA"
        )
    fallback_frames = min(362, 5 + 17 * max(0, round((total * 24 - 5) / 17)))
    try:
        effective_duration = float(
            document.get("effective_duration_seconds", fallback_frames / 24)
        )
    except (TypeError, ValueError) as error:
        raise ContractError("effective_duration_seconds must be numeric") from error
    if not math.isfinite(effective_duration) or not 5 / 24 <= effective_duration <= 362 / 24:
        raise ContractError("effective_duration_seconds is outside the H3 frame envelope")
    if abs(effective_duration - total) > 0.40:
        raise ContractError("effective duration does not match storyboard duration")
    reference_media: list[dict[str, str]] = []
    raw_media = document.get("reference_media", [])
    if raw_media is not None and not isinstance(raw_media, list):
        raise ContractError("reference_media must be a list")
    for index, item in enumerate(raw_media or (), 1):
        if not isinstance(item, dict):
            raise ContractError(f"reference_media {index} must be an object")
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in {"image", "video", "audio"}:
            raise ContractError(f"reference_media {index} has unsupported kind")
        reference_media.append({
            "kind": kind,
            "name": str(item.get("name", ""))[:300],
            "mime_type": str(item.get("mime_type", ""))[:100],
            "role": str(item.get("role", "")).strip()[:1000],
        })
    enhancement_scope = str(document.get("enhancement_scope", "full")).strip().lower()
    if enhancement_scope not in ENHANCEMENT_SCOPES:
        raise ContractError("enhancement_scope must be full, references, visuals or sound")
    if enhancement_scope == "references" and condition_mode != "REF2VA":
        raise ContractError("reference-object polishing requires the Ref2VA engine")
    raw_protocol = document.get("reference_protocol")
    reference_protocol = raw_protocol if isinstance(raw_protocol, dict) else None
    raw_soundtrack = document.get("soundtrack")
    soundtrack = None
    if isinstance(raw_soundtrack, dict):
        soundtrack = {
            "overall_soundscape": str(raw_soundtrack.get("overall_soundscape", ""))[:4000],
            "non_diegetic_music": str(raw_soundtrack.get("non_diegetic_music", ""))[:4000],
        }
    return EnhancementRequest(
        tuple(normalized), bgm_enabled, bgm_style, condition_mode,
        round(effective_duration, 6), tuple(reference_media), enhancement_scope,
        reference_protocol, soundtrack,
    )


def _image_part(content: bytes, mime_type: str, label: str) -> dict[str, Any]:
    encoded = base64.b64encode(content).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        "_label": label,
    }


def _audio_part(content: bytes, mime_type: str) -> dict[str, Any]:
    """Build MiMo's official base64 audio input part."""

    encoded = base64.b64encode(content).decode("ascii")
    return {
        "type": "input_audio",
        "input_audio": {"data": f"data:{mime_type};base64,{encoded}"},
    }


def _video_part(content: bytes, mime_type: str) -> dict[str, Any]:
    """Build MiMo V2.5's OpenAI-compatible native video input part.

    MiMo accepts a ``video_url`` data URL. Two FPS is sufficient for prompt
    editing while keeping the request and video-token cost bounded.
    """

    encoded = base64.b64encode(content).decode("ascii")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:{mime_type};base64,{encoded}"},
        "fps": 2,
        "media_resolution": "default",
    }


def _scope_instruction(request: EnhancementRequest) -> str:
    if request.condition_mode == "REF2VA":
        return ref2va_scope_instruction(request)
    return fl2va_scope_instruction(request)


def build_messages(
    request: EnhancementRequest,
    images: tuple[tuple[str, str, bytes], ...] = (),
    videos: tuple[tuple[str, str, bytes], ...] = (),
    audios: tuple[tuple[str, str, bytes], ...] = (),
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            _scope_instruction(request) + "\n"
            "Use the complete task below as context without changing its creative content. "
            "The first shot starts "
            "at 00:00; later shot timestamps are derived from durations. The conditioning "
            "method and "
            "effective duration below are authoritative.\n"
            + json.dumps({
                # Do not expose internal route names such as "Ref2VA" to the model.
                # They are implementation metadata and have previously leaked into
                # generated shot prose. The semantic method is all the editor needs.
                "conditioning_method": (
                    "full_reference"
                    if request.condition_mode == "REF2VA"
                    else request.condition_mode.lower()
                ),
                "effective_duration_seconds": request.effective_duration_seconds,
                "shots": list(request.shots),
                # Reference labels are compiler-owned control tokens. They are
                # repeated separately so the editor cannot mistake them for
                # optional prose that may be paraphrased away.
                "immutable_reference_bindings": [
                    {
                        "shot_id": shot["id"],
                        "audio_labels": [
                            f"<Audio {index}>"
                            for kind, index in REFERENCE_LABEL_RE.findall(shot["prompt"])
                            if kind.lower() == "audio"
                        ],
                    }
                    for shot in request.shots
                    if any(
                        kind.lower() == "audio"
                        for kind, _index in REFERENCE_LABEL_RE.findall(shot["prompt"])
                    )
                ],
                "soundtrack": {
                    "bgm_enabled": request.bgm_enabled,
                    "requested_bgm_style": request.bgm_style,
                    "current": request.soundtrack or {},
                },
                "current_reference_protocol": request.reference_protocol or {},
                "reference_media": list(request.reference_media),
            }, ensure_ascii=False)
        ),
    }]
    for label, mime_type, data in images:
        part = _image_part(data, mime_type, label)
        part.pop("_label")
        content.append({"type": "text", "text": f"The following image is the {label}."})
        content.append(part)
    for label, mime_type, data in videos:
        content.append({
            "type": "text",
            "text": (
                f"The following video is {label}. Inspect its visible subjects, "
                "motion, camera behavior and temporal continuity. Its embedded "
                "soundtrack is not an audio-reference binding."
            ),
        })
        content.append(_video_part(data, mime_type))
    for label, mime_type, data in audios:
        content.append({
            "type": "text",
            "text": (
                f"The following audio is {label}. Listen to it to identify its audible "
                "voice/timbre/delivery or environmental role. Do not copy its words."
            ),
        })
        content.append(_audio_part(data, mime_type))
    if request.condition_mode == "REF2VA" and (images or videos or audios):
        content.append({
            "type": "text",
            "text": (
                "Final editing priority: the user-authored shots and their requested "
                "result are the target. Do not caption, summarize, or recreate any "
                "reference media. Preserve the plot and exact dialogue, and return "
                f"only the requested {request.enhancement_scope} block."
            ),
        })
    return [
        {
            "role": "system",
            "content": (
                REF2VA_POLICY_PROMPT
                if request.condition_mode == "REF2VA"
                else FL2VA_SYSTEM_PROMPT
            ),
        },
        {"role": "user", "content": content},
    ]


def _strip_internal_mode_prefix(value: str) -> str:
    """Remove leaked route metadata from the beginning of creative prose.

    This is deliberately prefix-only: a user's legitimate story text elsewhere is
    left untouched, while accidental outputs such as ``Ref2VA模式，女孩走进房间``
    become ``女孩走进房间``.
    """
    cleaned = value.strip()
    while True:
        updated = INTERNAL_MODE_PREFIX_RE.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def _validated_reference_protocol(
    document: Any, request: EnhancementRequest
) -> dict[str, Any]:
    protocol = document.get("reference_protocol") if isinstance(document, dict) else None
    if not isinstance(protocol, dict):
        raise RuntimeError("MiMo omitted the Ref2VA reference protocol")

    definitions = protocol.get("subject_definitions")
    retention = protocol.get("retention_analysis")
    summary = str(protocol.get("summary", "")).strip()
    style_opening = str(protocol.get("style_opening", "")).strip()
    if not isinstance(definitions, list) or not all(
        isinstance(item, str) and item.strip() for item in definitions
    ):
        raise RuntimeError("MiMo returned invalid Ref2VA subject definitions")
    if not isinstance(retention, list) or not all(
        isinstance(item, str) and item.strip() for item in retention
    ):
        raise RuntimeError("MiMo returned invalid Ref2VA retention analysis")
    definitions = [item.strip() for item in definitions]
    retention = [item.strip() for item in retention]
    if len(definitions) > 24 or len(retention) > 36:
        raise RuntimeError("MiMo returned an oversized Ref2VA protocol")
    if not summary.startswith("[") or len(summary) > 4000:
        raise RuntimeError("MiMo returned an invalid Ref2VA summary")
    if len(style_opening) > 4000:
        raise RuntimeError("MiMo returned an oversized Ref2VA style opening")

    # This is an editorial response, not a compiler AST.  MiMo may add a short
    # explanatory definition or vary prose formatting without changing the
    # executable shot prompt.  Do not reject the whole enhancement for such
    # stylistic differences; only enforce bounded data and real media labels.
    media_counts = {
        "picture": sum(item["kind"] == "image" for item in request.reference_media),
        "video": sum(item["kind"] == "video" for item in request.reference_media),
        "audio": sum(item["kind"] == "audio" for item in request.reference_media),
    }
    combined = "\n".join((*definitions, summary, *retention, style_opening))
    for kind, raw_index in REFERENCE_LABEL_RE.findall(combined):
        index = int(raw_index)
        if index < 1 or index > media_counts[kind.lower()]:
            raise RuntimeError(f"MiMo referenced an unavailable <{kind} {index}>")
    return {
        "subject_definitions": definitions,
        "summary": summary,
        "retention_analysis": retention,
        "style_opening": style_opening,
    }


def validate_response(document: Any, request: EnhancementRequest) -> dict[str, Any]:
    if request.enhancement_scope != "full":
        if not isinstance(document, dict):
            raise RuntimeError("MiMo returned an invalid scoped result")
        base_protocol = dict(request.reference_protocol or {})
        returned_protocol = document.get("reference_protocol")
        if isinstance(returned_protocol, dict):
            base_protocol.update(returned_protocol)
        merged = {
            "reference_protocol": base_protocol,
            "shots": document.get("shots", list(request.shots)),
            "soundtrack": document.get("soundtrack", request.soundtrack or {}),
        }
        full = validate_response(merged, replace(request, enhancement_scope="full"))
        if request.enhancement_scope == "references":
            protocol = full["reference_protocol"]
            return {"reference_protocol": {
                "subject_definitions": protocol["subject_definitions"],
                "retention_analysis": protocol["retention_analysis"],
            }}
        if request.enhancement_scope == "visuals":
            result: dict[str, Any] = {"shots": full["shots"]}
            if request.condition_mode == "REF2VA":
                protocol = full["reference_protocol"]
                result["reference_protocol"] = {
                    "summary": protocol["summary"],
                }
            return result
        return {"soundtrack": full["soundtrack"]}
    if not isinstance(document, dict) or not isinstance(document.get("shots"), list):
        raise RuntimeError("MiMo returned an invalid storyboard")
    shots = document["shots"]
    if len(shots) != len(request.shots):
        raise RuntimeError("MiMo changed the number of shots")
    normalized: list[dict[str, Any]] = []
    for source, enhanced in zip(request.shots, shots):
        if not isinstance(enhanced, dict):
            raise RuntimeError("MiMo returned an invalid shot")
        prompt = _strip_internal_mode_prefix(str(enhanced.get("prompt", "")))
        if not prompt or len(prompt) > MAX_SHOT_TEXT:
            raise RuntimeError("MiMo returned an empty or oversized shot")
        if any(field in prompt.lower() for field in TOP_LEVEL_H3_FIELDS):
            raise RuntimeError("MiMo placed top-level H3 syntax inside a shot")
        source_audio_labels = {
            int(index)
            for kind, index in REFERENCE_LABEL_RE.findall(str(source["prompt"]))
            if kind.lower() == "audio"
        }
        enhanced_audio_labels = {
            int(index)
            for kind, index in REFERENCE_LABEL_RE.findall(prompt)
            if kind.lower() == "audio"
        }
        if not source_audio_labels.issubset(enhanced_audio_labels):
            # Creative prose belongs to MiMo; executable reference bindings do
            # not. If the editor drops a binding, preserve the user's complete
            # source shot rather than emitting a broken prompt or asking the
            # user to retry a nondeterministic model response.
            prompt = str(source["prompt"]).strip()
        normalized.append({
            "id": source["id"],
            # Durations are owned by the deterministic editor, never by the LLM.
            "duration_seconds": source["duration_seconds"],
            "prompt": prompt,
        })
    soundtrack = document.get("soundtrack")
    if not isinstance(soundtrack, dict):
        soundtrack = {}
    music = str(soundtrack.get("non_diegetic_music", "")).strip()
    if not request.bgm_enabled:
        music = "N/A"
    elif not music:
        music = request.bgm_style
    result = {
        "shots": normalized,
        "soundtrack": {
            "overall_soundscape": str(soundtrack.get("overall_soundscape", "")).strip(),
            "non_diegetic_music": music,
        },
    }
    if request.condition_mode == "REF2VA":
        result["reference_protocol"] = _validated_reference_protocol(document, request)
    return result


def _mimo_audit_directory() -> Path:
    configured = os.environ.get("H3_SERVE_MIMO_AUDIT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().absolute()
    return Path(__file__).resolve().parents[1] / "runtime" / "mimo-audit"


def _write_mimo_audit(path: Path, name: str, value: Any) -> None:
    """Best-effort local audit logging; logging must never break polishing."""

    try:
        path.mkdir(parents=True, exist_ok=True)
        target = path / name
        if isinstance(value, str):
            target.write_text(value, encoding="utf-8")
        else:
            target.write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except OSError:
        pass


class MiMoPromptEnhancer:
    async def _complete(
        self,
        *,
        api_key: str,
        messages: list[dict[str, Any]],
        max_completion_tokens: int,
        temperature: float,
    ) -> Any:
        if not api_key.strip():
            raise ContractError("请先在设置中填写 MiMo API Key")
        payload = {
            "model": MIMO_MODEL,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
        }
        audit_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "_" + uuid.uuid4().hex[:8]
        )
        audit_path = _mimo_audit_directory() / audit_id
        # payload contains the exact text and base64 media sent to MiMo, but no
        # Authorization header or API key.
        _write_mimo_audit(audit_path, "request.json", payload)
        timeout = aiohttp.ClientTimeout(total=120, connect=15)
        headers = {"Authorization": f"Bearer {api_key.strip()}"}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(MIMO_ENDPOINT, json=payload, headers=headers) as response:
                    body = await response.text()
                    _write_mimo_audit(audit_path, "response.txt", body)
                    _write_mimo_audit(audit_path, "http.json", {
                        "status": response.status,
                        "endpoint": MIMO_ENDPOINT,
                    })
                    if response.status >= 400:
                        # Never echo the authorization header or a large provider response.
                        detail = re.sub(r"\s+", " ", body)[:300]
                        raise RuntimeError(f"MiMo API 请求失败（HTTP {response.status}）：{detail}")
        except asyncio.TimeoutError as error:
            _write_mimo_audit(audit_path, "error.txt", "timeout")
            raise RuntimeError("MiMo API 请求超时，请稍后重试") from error
        except aiohttp.ClientError as error:
            _write_mimo_audit(audit_path, "error.txt", repr(error))
            raise RuntimeError("无法连接小米 MiMo API，请检查网络后重试") from error
        try:
            envelope = json.loads(body)
            content = envelope["choices"][0]["message"]["content"]
            document = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("MiMo API 返回了无法解析的结构化结果") from error
        return document

    async def enhance(
        self,
        *,
        api_key: str,
        request: EnhancementRequest,
        images: tuple[tuple[str, str, bytes], ...] = (),
        videos: tuple[tuple[str, str, bytes], ...] = (),
        audios: tuple[tuple[str, str, bytes], ...] = (),
    ) -> dict[str, Any]:
        document = await self._complete(
            api_key=api_key,
            messages=build_messages(request, images, videos, audios),
            max_completion_tokens=6144 if request.condition_mode == "REF2VA" else 4096,
            temperature=0.2,
        )
        return validate_response(document, request)
