from __future__ import annotations

import json
import unittest

from h3serve.contract import ContractError
from h3serve.prompt_enhancer import (
    build_messages,
    parse_enhancement_request,
    validate_response,
)


class PromptEnhancerContractTest(unittest.TestCase):
    def request(self, *, bgm: bool = False):
        return parse_enhancement_request(json.dumps({
            "shots": [
                {"id": "a", "duration_seconds": 2, "prompt": "女孩推门并说：‘我回来了。’"},
                {"id": "b", "duration_seconds": 3, "prompt": "镜头跟随她走进房间。"},
            ],
            "bgm_enabled": bgm,
            "bgm_style": "克制的弦乐" if bgm else "",
            "condition_mode": "FL2VA",
            "effective_duration_seconds": 124 / 24,
        }, ensure_ascii=False))

    def test_bgm_off_is_a_hard_postcondition(self) -> None:
        request = self.request()
        result = validate_response({
            "shots": [
                {"id": "wrong", "duration_seconds": 9,
                 "prompt": "女孩说：<d>[Chinese] 我回来了。</d>"},
                {"id": "wrong2", "duration_seconds": 9, "prompt": "增强二"},
            ],
            "soundtrack": {
                "overall_soundscape": "门响和脚步声",
                "non_diegetic_music": "激昂交响乐",
            },
        }, request)
        self.assertEqual(result["soundtrack"]["non_diegetic_music"], "N/A")
        self.assertEqual([item["id"] for item in result["shots"]], ["a", "b"])
        self.assertEqual(
            [item["duration_seconds"] for item in result["shots"]], [2.0, 3.0]
        )

    def test_model_cannot_change_shot_count(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "number of shots"):
            validate_response({"shots": [{"prompt": "only one"}]}, self.request())

    def test_model_cannot_embed_top_level_fields_inside_a_shot(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "top-level H3 syntax"):
            validate_response({
                "shots": [{"prompt": "integrated_multimodal_description: 我回来了。"},
                          {"prompt": "镜头跟随她走进房间。"}],
                "soundtrack": {},
            }, self.request())

    def test_total_duration_and_bgm_style_are_validated(self) -> None:
        with self.assertRaises(ContractError):
            parse_enhancement_request(json.dumps({
                "shots": [{"duration_seconds": 10, "prompt": "a"},
                          {"duration_seconds": 10, "prompt": "b"}],
            }))
        with self.assertRaisesRegex(ContractError, "BGM style"):
            parse_enhancement_request(json.dumps({
                "shots": [{"duration_seconds": 5, "prompt": "a"}],
                "bgm_enabled": True,
            }))

    def test_messages_include_rules_and_optional_images(self) -> None:
        messages = build_messages(
            self.request(), (("opening frame", "image/png", b"png"),)
        )
        self.assertIn("non_diegetic_music", messages[0]["content"])
        self.assertIn("stable (S1), (S2)", messages[0]["content"])
        self.assertIn("Disabling BGM never removes authored dialogue", messages[0]["content"])
        self.assertIn("Never translate", messages[0]["content"])
        self.assertIn("Never invent a plot event", messages[0]["content"])
        self.assertIn("Do not repeat", messages[0]["content"])
        self.assertIn("never emit subject_definitions", messages[0]["content"])
        self.assertNotIn("Ref2VA", messages[0]["content"])
        content = messages[1]["content"]
        self.assertIn('"conditioning_method": "fl2va"', content[0]["text"])
        self.assertIn('"effective_duration_seconds"', content[0]["text"])
        self.assertEqual(content[-1]["type"], "image_url")
        self.assertTrue(content[-1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_messages_include_native_mimo_video_input(self) -> None:
        messages = build_messages(
            self.request(), videos=(("<Video 1>", "video/mp4", b"mp4"),)
        )
        content = messages[1]["content"]
        video = next(part for part in content if part["type"] == "video_url")
        self.assertTrue(video["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertEqual(video["fps"], 2)

    def test_ref2va_storyboard_is_prioritized_over_reference_captioning(self) -> None:
        request = parse_enhancement_request(json.dumps({
            "condition_mode": "REF2VA", "effective_duration_seconds": 5,
            "shots": [{"id": "one", "duration_seconds": 5,
                       "prompt": "将<Video 1>中的物体改成目标版本。"}],
            "reference_media": [{"kind": "video", "name": "clip.mp4",
                                 "mime_type": "video/mp4", "role": "reference"}],
        }))
        messages = build_messages(request, videos=(("<Video 1>", "video/mp4", b"mp4"),))
        self.assertIn("sole target specification", messages[0]["content"])
        self.assertIn("not an alternative story", messages[0]["content"])
        self.assertIn("Do not caption, summarize, or recreate", messages[1]["content"][-1]["text"])

    def test_polish_rejects_blank_shots(self) -> None:
        with self.assertRaisesRegex(ContractError, "written before polishing"):
            parse_enhancement_request(json.dumps({
                "shots": [{"duration_seconds": 5, "prompt": ""}],
            }))

    def test_ref2va_editor_does_not_fail_on_noncanonical_definition_prose(self) -> None:
        request = parse_enhancement_request(json.dumps({
            "shots": [{"duration_seconds": 5, "prompt": "人物走进房间。"}],
            "condition_mode": "REF2VA",
            "effective_duration_seconds": 124 / 24,
            "reference_media": [
                {"kind": "image", "name": "person.png", "mime_type": "image/png"},
                {"kind": "audio", "name": "voice.wav", "mime_type": "audio/wav"},
            ],
        }, ensure_ascii=False))
        base = {
            "reference_protocol": {
                "subject_definitions": [
                    "<Subject 1> is the person from <Picture 1>.",
                    "<Audio 1> is a generic voice reference.",
                ],
                "summary": "[reference generation + audio reference] A room scene.",
                "retention_analysis": [
                    "<Picture 1>: attribute_transfer — preserve identity.",
                    "<Audio 1>: reference — preserve timbre.",
                ],
                "style_opening": "Maintain continuity.",
            },
            "shots": [{"prompt": "The person enters the room."}],
            "soundtrack": {},
        }
        result = validate_response(base, request)
        self.assertEqual(result["shots"][0]["prompt"], "The person enters the room.")
        self.assertIn(
            "<Audio 1> is a generic voice reference.",
            result["reference_protocol"]["subject_definitions"],
        )

    def test_reference_media_metadata_is_preserved_for_enhancement(self) -> None:
        request = parse_enhancement_request(json.dumps({
            "shots": [{"duration_seconds": 5, "prompt": "参考人物进门。"}],
            "condition_mode": "REF2VA",
            "effective_duration_seconds": 124 / 24,
            "reference_media": [
                {"kind": "image", "name": "face.png", "mime_type": "image/png",
                 "role": "女孩(S1)的人物身份参考"},
                {"kind": "audio", "name": "voice.wav", "mime_type": "audio/wav",
                 "role": "女孩(S1)的音色参考，不复用台词"},
            ],
        }, ensure_ascii=False))
        messages = build_messages(request)
        text = messages[1]["content"][0]["text"]
        self.assertIn('"name": "face.png"', text)
        self.assertIn('"name": "voice.wav"', text)
        self.assertIn("女孩(S1)的音色参考", text)

    def test_audio_bindings_are_sent_as_immutable_editor_constraints(self) -> None:
        request = parse_enhancement_request(json.dumps({
            "shots": [{"id": "voice", "duration_seconds": 5,
                       "prompt": "男人(S1)用 <Audio 2> 的音色说：‘出发。’"}],
            "condition_mode": "REF2VA",
            "effective_duration_seconds": 124 / 24,
            "reference_media": [
                {"kind": "image", "name": "man.png", "mime_type": "image/png"},
                {"kind": "audio", "name": "a.wav", "mime_type": "audio/wav"},
                {"kind": "audio", "name": "b.wav", "mime_type": "audio/wav"},
            ],
        }, ensure_ascii=False))
        text = build_messages(request)[1]["content"][0]["text"]
        self.assertIn('"immutable_reference_bindings"', text)
        self.assertIn('"audio_labels": ["<Audio 2>"]', text)

    def test_ref2va_mode_is_supported_without_uploading_local_references(self) -> None:
        request = parse_enhancement_request(json.dumps({
            "shots": [{"duration_seconds": 5, "prompt": "参考人物走进大厅。"}],
            "condition_mode": "Ref2VA",
            "effective_duration_seconds": 124 / 24,
        }, ensure_ascii=False))
        self.assertEqual(request.condition_mode, "REF2VA")
        messages = build_messages(request)
        self.assertIn(
            '"conditioning_method": "full_reference"',
            messages[1]["content"][0]["text"],
        )
        self.assertNotIn('"condition_mode": "REF2VA"', messages[1]["content"][0]["text"])
        self.assertIn("official six-section", messages[0]["content"])
        self.assertIn("Put only exact spoken words", messages[0]["content"])
        self.assertIn("never append a", messages[0]["content"])
        self.assertNotIn("English production prose with exact original dialogue", messages[0]["content"])
        self.assertIn("Never write model names", messages[0]["content"])
        self.assertEqual(len(messages[1]["content"]), 1)

    def test_fl2va_and_ref2va_editor_policies_are_independent(self) -> None:
        fl = build_messages(self.request())[0]["content"]
        ref_request = parse_enhancement_request(json.dumps({
            "shots": [{"duration_seconds": 5, "prompt": "Use <Picture 1>."}],
            "condition_mode": "REF2VA",
            "effective_duration_seconds": 124 / 24,
            "reference_media": [{"kind": "image", "name": "a.png"}],
        }))
        ref = build_messages(ref_request)[0]["content"]
        self.assertIn("opening frame", fl)
        self.assertIn("never emit subject_definitions, retention_analysis", fl)
        self.assertIn("retention_analysis", ref)
        self.assertNotIn("endpoint-alignment", fl)
        self.assertIn("Never use first-frame", ref)

    def test_internal_mode_prefix_is_removed_from_ref2va_shot(self) -> None:
        request = parse_enhancement_request(json.dumps({
            "shots": [{"id": "a", "duration_seconds": 5,
                       "prompt": "女孩走进房间。"}],
            "condition_mode": "REF2VA",
            "effective_duration_seconds": 124 / 24,
            "reference_media": [
                {"kind": "image", "name": "girl.png", "mime_type": "image/png"},
            ],
        }, ensure_ascii=False))
        result = validate_response({
            "reference_protocol": {
                "subject_definitions": ["<Subject 1> is the girl from <Picture 1>."],
                "summary": "[reference generation] A room scene.",
                "retention_analysis": [
                    "<Picture 1>: attribute_transfer — preserve identity."
                ],
                "style_opening": "One continuous take.",
            },
            "shots": [{"prompt": "Ref2VA模式，女孩走进房间。"}],
            "soundtrack": {},
        }, request)
        self.assertEqual(result["shots"][0]["prompt"], "女孩走进房间。")

    def test_ref2va_response_requires_and_returns_six_section_protocol(self) -> None:
        request = parse_enhancement_request(json.dumps({
            "shots": [{"id": "a", "duration_seconds": 5,
                       "prompt": "女孩说：‘快走！’"}],
            "condition_mode": "REF2VA",
            "effective_duration_seconds": 124 / 24,
            "reference_media": [
                {"kind": "image", "name": "girl.png", "mime_type": "image/png",
                 "role": "女孩(S1)身份"},
                {"kind": "audio", "name": "girl.wav", "mime_type": "audio/wav",
                 "role": "女孩(S1)音色"},
            ],
        }, ensure_ascii=False))
        result = validate_response({
            "reference_protocol": {
                "subject_definitions": [
                    "<Subject 1> is the girl (S1), derived from <Picture 1>.",
                    "<Audio 1> is the voice-timbre reference for <Subject 1> (S1).",
                ],
                "summary": "[reference generation + audio reference] A tense dialogue scene.",
                "retention_analysis": [
                    "<Picture 1>: attribute_transfer — preserve identity.",
                    "<Audio 1>: reference — voice timbre for <Subject 1> (S1).",
                ],
                "style_opening": "One continuous cinematic take.",
            },
            "shots": [{"prompt": "The girl <Subject 1> (S1) uses <Audio 1> and says <d>[Chinese] 快走！</d>."}],
            "soundtrack": {
                "overall_soundscape": "Footsteps and clean dialogue.",
                "non_diegetic_music": "music that must be removed",
            },
        }, request)
        self.assertEqual(result["reference_protocol"]["summary"],
                         "[reference generation + audio reference] A tense dialogue scene.")
        self.assertEqual(result["soundtrack"]["non_diegetic_music"], "N/A")

    def test_ref2va_protocol_cannot_invent_reference_labels(self) -> None:
        request = parse_enhancement_request(json.dumps({
            "shots": [{"duration_seconds": 5, "prompt": "人物走进房间。"}],
            "condition_mode": "REF2VA",
            "effective_duration_seconds": 124 / 24,
            "reference_media": [
                {"kind": "image", "name": "one.png", "mime_type": "image/png"},
            ],
        }, ensure_ascii=False))
        with self.assertRaisesRegex(RuntimeError, "unavailable <Picture 2>"):
            validate_response({
                "reference_protocol": {
                    "subject_definitions": ["<Subject 1> is from <Picture 2>."],
                    "summary": "[reference generation] A room scene.",
                    "retention_analysis": [
                        "<Picture 2>: attribute_transfer — preserve identity."
                    ],
                    "style_opening": "Stable framing.",
                },
                "shots": [{"prompt": "The subject enters the room."}],
                "soundtrack": {},
            }, request)

    def test_ref2va_enhancer_cannot_drop_audio_binding(self) -> None:
        request = parse_enhancement_request(json.dumps({
            "shots": [{"id": "a", "duration_seconds": 5,
                       "prompt": "<Subject 1> (S1) 使用 <Audio 1> 的音色说：‘快走！’"}],
            "condition_mode": "REF2VA",
            "effective_duration_seconds": 124 / 24,
            "reference_media": [
                {"kind": "image", "name": "man.png", "mime_type": "image/png"},
                {"kind": "audio", "name": "man.wav", "mime_type": "audio/wav"},
            ],
        }, ensure_ascii=False))
        document = {
            "reference_protocol": {
                "subject_definitions": [
                    "<Subject 1> is the man from <Picture 1>.",
                    "<Audio 1> is the voice-timbre reference for <Subject 1> (S1).",
                ],
                "summary": "[reference generation + audio reference] A warning.",
                "retention_analysis": [
                    "<Picture 1>: attribute_transfer — preserve identity.",
                    "<Audio 1>: reference — preserve voice timbre and delivery.",
                ],
                "style_opening": "One continuous take.",
            },
            "shots": [{"prompt": "<Subject 1> (S1) says <d>[Chinese] 快走！</d>."}],
            "soundtrack": {},
        }
        result = validate_response(document, request)
        self.assertEqual(result["shots"][0]["prompt"], request.shots[0]["prompt"])

    def test_effective_duration_and_mode_are_validated(self) -> None:
        with self.assertRaisesRegex(ContractError, "condition_mode"):
            parse_enhancement_request(json.dumps({
                "shots": [{"duration_seconds": 5, "prompt": "a"}],
                "condition_mode": "unknown",
            }))
        with self.assertRaisesRegex(ContractError, "does not match"):
            parse_enhancement_request(json.dumps({
                "shots": [{"duration_seconds": 5, "prompt": "a"}],
                "effective_duration_seconds": 10,
            }))

    def scoped_ref_request(self, scope: str):
        return parse_enhancement_request(json.dumps({
            "shots": [{"id": "one", "duration_seconds": 5,
                       "prompt": "<Subject 1>走进房间。"}],
            "condition_mode": "REF2VA",
            "effective_duration_seconds": 124 / 24,
            "enhancement_scope": scope,
            "reference_media": [
                {"kind": "image", "name": "girl.png", "mime_type": "image/png"},
            ],
            "reference_protocol": {
                "subject_definitions": ["<Subject 1> is the girl in <Picture 1>."],
                "summary": "[reference generation] A girl enters a room.",
                "retention_analysis": [
                    "<Subject 1>: fully_preserved - preserve the girl's identity."
                ],
                "style_opening": "A realistic continuous shot.",
            },
            "soundtrack": {
                "overall_soundscape": "Quiet room ambience.",
                "non_diegetic_music": "N/A",
            },
        }, ensure_ascii=False))

    def test_reference_scope_returns_only_reference_contract(self) -> None:
        request = self.scoped_ref_request("references")
        messages = build_messages(request)
        self.assertIn("Polish only the reference-object contract", messages[1]["content"][0]["text"])
        self.assertIn("fully_preserved, partially_preserved", messages[1]["content"][0]["text"])
        result = validate_response({
            "reference_protocol": {
                "subject_definitions": [
                    "<Subject 1> is the young girl in <Picture 1>."
                ],
                "retention_analysis": [
                    "<Subject 1>: fully_preserved - preserve identity and clothing."
                ],
            }
        }, request)
        self.assertEqual(set(result), {"reference_protocol"})
        self.assertNotIn("summary", result["reference_protocol"])

    def test_visual_scope_preserves_duration_and_returns_no_soundtrack(self) -> None:
        request = self.scoped_ref_request("visuals")
        instruction = build_messages(request)[1]["content"][0]["text"]
        self.assertIn("Do not invent dialogue, narration, singing, filler syllables", instruction)
        self.assertIn("Make actions and dialogue plausibly fit each shot's duration", instruction)
        result = validate_response({
            "reference_protocol": {
                "summary": "[reference generation] A girl enters a quiet room.",
            },
            "shots": [{"id": "changed", "duration_seconds": 99,
                       "prompt": "<Subject 1> enters the room."}],
        }, request)
        self.assertEqual(result["shots"][0]["duration_seconds"], 5.0)
        self.assertEqual(set(result), {"shots", "reference_protocol"})
        self.assertEqual(set(result["reference_protocol"]), {"summary"})

    def test_sound_scope_returns_only_soundtrack_and_forces_bgm_off(self) -> None:
        request = self.scoped_ref_request("sound")
        instruction = build_messages(request)[1]["content"][0]["text"]
        self.assertIn("only ambience, physical action sounds", instruction)
        self.assertIn("dialogue, singing, and diegetic music", instruction)
        self.assertIn("instrumentation, tempo, rhythm, and dynamic change", instruction)
        result = validate_response({
            "soundtrack": {
                "overall_soundscape": "Soft footsteps in a quiet room.",
                "non_diegetic_music": "Loud orchestra.",
            }
        }, request)
        self.assertEqual(set(result), {"soundtrack"})
        self.assertEqual(result["soundtrack"]["non_diegetic_music"], "N/A")


if __name__ == "__main__":
    unittest.main()
