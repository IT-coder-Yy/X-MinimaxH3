#!/usr/bin/env python3
"""Replay historical V24 calibrations for an offline Human-review batch.

The shared cases exercise points whose V24 surface is identical for every
historical calibration.  The comparison cases replay V009/C01/C02/C03 through
the explicitly named research compiler.  The production service never imports
this choice into its request path and remains fixed to the final C02 surface.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h3serve.config import ServicePaths
from h3serve.contract import GenerationSpec
from h3serve.native_engine import NativeHotH3Engine
from h3serve.native_engine.planner import V24ResearchParetoRuntimeSelector
from h3serve.native_engine.session_factory import NativeSessionFactory, NativeSessionPaths


@dataclass(frozen=True, slots=True)
class ReviewCase:
    case_id: str
    resolution: str
    duration_seconds: float
    seed: int
    prompt: str
    candidates: tuple[str, ...] = ("c03",)


RESEARCH_CALIBRATION_IDS = {
    "stable": "v24_final_stable_v009",
    "c01": "v24_final_c01_v014b_shield_u7p00",
    "c02": "v24_final_c02_round2_trajectory_u7p00",
    "c03": "v24_final_c03_round2_trajectory_u7p30",
}


SHARED_CASES = (
    ReviewCase(
        case_id="shared_480p6_watch_case_contact",
        resolution="480p",
        duration_seconds=6,
        seed=82601,
        prompt=(
            "6秒写实电影短片，固定中景镜头。安静的钟表维修台前，一位戴圆框眼镜的女钟表匠"
            "用左手扶住一只打开的黄铜怀表盒，右手把细小齿轮放回盒内；她的手指明确接触表盖后，"
            "才把表盖平稳合上。桌上的镊子、螺丝盒和后方挂钟始终保持原位、形状和数量不变。"
            "动作连续自然，手指与物体边缘清晰，无自动移动、无残影、无镜头切换。保留轻微金属碰触声和室内环境声，不要对白，不要背景音乐。"
        ),
    ),
    ReviewCase(
        case_id="shared_480p12_ceramic_handoff",
        resolution="480p",
        duration_seconds=12,
        seed=82602,
        prompt=(
            "12秒写实电影短片，固定中景镜头。明亮的陶艺工作室里，女陶艺师双手托住一只带蓝色花纹的白瓷杯，"
            "走到木桌另一侧，把杯子递给男学徒；男学徒的双手先稳稳接触杯身和杯底，女陶艺师随后才松手。"
            "男学徒把杯子直立放在软布中央，说：\"釉面完好，我会小心收好。\""
            "背景架上的三只陶罐、桌边工具和窗框全程稳定，不增减、不漂移、不形变。交接因果明确，手部和杯沿清晰，"
            "口型自然同步，无闪烁亮斑、无镜头切换。保留真实脚步声、瓷器轻响和室内环境声，不要背景音乐。"
        ),
    ),
    ReviewCase(
        case_id="shared_480p15_archive_stamp",
        resolution="480p",
        duration_seconds=15,
        seed=82603,
        prompt=(
            "15秒写实电影短片，固定中景镜头。旧火车站的档案室里，男档案员从桌面拿起一张米黄色登记卡，"
            "把卡片平放在绿色桌垫上；女主管扶住卡片左侧，男档案员拿起木柄印章，垂直按下后再抬起，卡片上留下一个红色圆章。"
            "女主管看清印记后说：\"编号正确，可以归档。\"男档案员点头，把卡片插入标有A17的文件夹。"
            "桌上的台灯、墨盒、电话和后方文件柜始终固定，文字牌和物体结构不突变。手与卡片接触清楚，印章动作符合物理，"
            "口型语速自然，无背景扭曲、无闪烁、无镜头切换。保留纸张摩擦、印章轻响和房间环境声，不要背景音乐。"
        ),
    ),
    ReviewCase(
        case_id="shared_720p7_teapot_pour",
        resolution="720p",
        duration_seconds=7,
        seed=82604,
        prompt=(
            "7秒写实电影短片，固定中近景镜头。安静的茶室里，一位男服务员左手扶住白瓷茶壶盖，右手握住壶柄，"
            "缓慢倾斜茶壶，把一股连续清晰的茶水倒进桌上的单只蓝边茶杯，然后把茶壶平稳放回竹垫。"
            "壶嘴始终对准杯口，液体不穿模、不溢出；茶杯、竹垫和背景木格窗全程稳定。手指、壶柄和杯沿清晰，"
            "动作自然，无物体自行移动、无残影、无镜头切换。保留倒水声和轻微室内环境声，不要对白，不要背景音乐。"
        ),
    ),
)


DISCRIMINATIVE_CASE = ReviewCase(
    case_id="compare_720p12_clockwork_gear_handoff",
    resolution="720p",
    duration_seconds=12,
    seed=82612,
    candidates=("stable", "c01", "c02", "c03"),
    prompt=(
        "12秒写实电影短片，固定中近景镜头。博物馆修复室里，一只打开的古董座钟平放在铺有深蓝软布的工作台上。"
        "女修复师用镊子夹住一枚小铜齿轮，把它递向男修复师；男修复师先用拇指和食指稳稳接触齿轮，女修复师随后才松开镊子。"
        "他把齿轮垂直放入钟表机芯中央的空轴，轻轻旋转确认啮合。女修复师说：\"位置对了，先别合上外壳。\""
        "座钟外壳保持打开且静止，机芯结构、齿轮数量、桌上三件工具和背景玻璃柜全程不增减、不漂移、不形变。"
        "交接和安装的接触顺序明确，手指、镊子、齿轮边缘与嘴型清晰，语速自然；无自动关合、无残影、无背景波动、"
        "无闪烁亮斑、无镜头切换。保留金属轻响、布料摩擦和安静室内环境声，不要背景音乐。"
    ),
)


def _compiled_h3_prompt(*, shot_body: str, soundscape: str) -> str:
    """Serialize an already edited FL-family shot with the UI's H3 contract."""

    return (
        "integrated_multimodal_description: [Shot 1]\n"
        + shot_body.strip()
        + "\n\n"
        + "overall_soundscape: "
        + soundscape.strip()
        + "\n\nnon_diegetic_music: N/A"
    )


STRUCTURED_SHARED_CASES = (
    ReviewCase(
        case_id="h3_structured_shared_480p12_ceramic_handoff",
        resolution="480p",
        duration_seconds=12,
        seed=82702,
        prompt=_compiled_h3_prompt(
            shot_body="""
A realistic cinematic continuous medium shot inside a bright ceramic studio. The same
female ceramic artist (S1), wearing a beige apron, stands on the left side of a wooden
worktable. The same male apprentice (S2), wearing a dark green apron, stands on the
right. A single white porcelain cup with a blue floral glaze rests in S1's two hands.

During the opening four seconds, S1 walks around the near corner of the table while
keeping the cup upright and clearly visible. From four to eight seconds, she presents
the cup across the table. S2 places one hand around the cup body and the other beneath
its base; only after both hands have secure contact does S1 release it. From eight
seconds to the end, S2 places the same cup upright on the center of a folded cloth,
looks at S1, and says naturally in Mandarin:
<d>[Chinese] 釉面完好，我会小心收好。</d>
S1 watches silently with relaxed closed lips. The camera remains steady at medium
distance, preserving the hand-to-cup contact and the cup's blue pattern throughout the
transfer. Three finished pots on the rear shelf remain part of the stable composition.
""",
            soundscape=(
                "Quiet ceramic-studio room tone, soft footsteps, apron movement, a delicate "
                "porcelain contact on cloth, and natural close-room reverberation."
            ),
        ),
    ),
    ReviewCase(
        case_id="h3_structured_shared_480p15_archive_stamp",
        resolution="480p",
        duration_seconds=15,
        seed=82703,
        prompt=_compiled_h3_prompt(
            shot_body="""
A realistic cinematic continuous medium shot in the archive room of an old railway
station. The same male archivist (S1), wearing a brown waistcoat, works at the right
side of a green desk mat. The same female supervisor (S2), wearing a navy cardigan,
stands to his left. A cream registration card, a wooden-handled stamp, a red ink pad,
and a folder marked A17 are clearly arranged on the desk.

During the opening five seconds, S1 lifts the registration card, aligns it flat on the
green mat, and keeps one hand near its lower edge. From five to ten seconds, S2 places
two fingertips on the card's left edge to hold it steady. S1 inks the stamp, presses it
straight down onto the card, then lifts it to reveal one red circular mark. S2 examines
the result and says calmly in Mandarin:
<d>[Chinese] 编号正确，可以归档。</d>
S1 listens silently. From ten seconds to the end, S1 opens the A17 folder and slides
the same stamped card fully inside. The camera holds a stable medium composition so
the stamp contact, the resulting mark, and the card entering the folder remain visible.
The desk lamp, black telephone, and rear filing cabinets retain their established
positions through the take.
""",
            soundscape=(
                "Low archive-room ambience, paper sliding across felt, a stamp touching the "
                "ink pad and card, a folder opening, and subtle clothing movement."
            ),
        ),
    ),
    ReviewCase(
        case_id="h3_structured_shared_720p7_teapot_pour",
        resolution="720p",
        duration_seconds=7,
        seed=82704,
        prompt=_compiled_h3_prompt(
            shot_body="""
A realistic cinematic continuous medium-close shot in a quiet tea room. The same male
server, wearing a charcoal linen jacket, stands behind a low wooden table. A single
white porcelain teapot sits on a round bamboo mat beside a single blue-rimmed teacup.

He places his left fingertips on the teapot lid and closes his right hand around the
handle. He lifts the pot, tilts the spout above the cup, and pours one continuous stream
of amber tea into the cup. After the liquid level rises, he returns the pot upright and
sets it back on the same bamboo mat. The camera stays fixed at medium-close distance,
keeping his fingers, the handle, the spout, the liquid stream, and the cup rim in clear
view. The wooden lattice window remains a calm background element. No one speaks.
""",
            soundscape=(
                "A clear continuous tea-pouring sound, a gentle porcelain contact on bamboo, "
                "soft sleeve movement, and quiet tea-room ambience without human voices."
            ),
        ),
    ),
)


STRUCTURED_COMPARISON_CASES = (
    ReviewCase(
        case_id="h3_structured_compare_720p12_clockwork_handoff",
        resolution="720p",
        duration_seconds=12,
        seed=82712,
        candidates=("c01", "c02", "c03"),
        prompt=_compiled_h3_prompt(
            shot_body="""
A realistic cinematic continuous medium-close shot inside a museum conservation room.
The same female conservator (S1), wearing a light gray coat, stands on the left side of
a worktable. The same male conservator (S2), wearing a dark blue coat, stands on the
right. An open antique mantel clock rests on deep-blue cloth between them, revealing a
single empty spindle in its brass movement.

During the opening four seconds, S1 lifts one small copper gear with steel tweezers and
holds it above the cloth. From four to eight seconds, she brings the gear toward S2.
S2 first secures the gear between his thumb and forefinger; S1 opens the tweezers only
after his fingers make clear contact. From eight seconds to the end, S2 lowers the same
gear vertically onto the empty spindle and turns it gently until it engages the adjacent
gear. S1 watches the mechanism and says in Mandarin:
<d>[Chinese] 位置对了，先别合上外壳。</d>
S2 remains silent with closed lips. The camera stays steady and close enough to preserve
the handoff, the gear edges, and the clock movement. The clock case remains open through
the final frame, while three hand tools and the glass cabinet behind them maintain the
established composition.
""",
            soundscape=(
                "Quiet conservation-room ambience, faint tweezers and brass contacts, soft "
                "cloth movement, and subtle handling sounds from the clock mechanism."
            ),
        ),
    ),
    ReviewCase(
        case_id="h3_structured_compare_720p14_bookbindery_tool_handoff",
        resolution="720p",
        duration_seconds=14,
        seed=82714,
        candidates=("c01", "c02", "c03"),
        prompt=_compiled_h3_prompt(
            shot_body="""
A realistic cinematic continuous medium shot in a traditional bookbinding workshop.
The same female bookbinder (S1), wearing a rust-colored apron, stands at the left end of
a long workbench. The same male assistant (S2), wearing a slate-blue shirt, stands at
the right. One half-bound book with a blue linen cover lies open at the center, beside
a single ivory-colored bone folder.

During the opening five seconds, S1 folds the blue linen smoothly over the near edge of
the book and holds the fold with her left palm. From five to nine seconds, she picks up
the bone folder and offers its handle to S2. S2 closes his right hand around the handle
before S1 releases it. From nine seconds to the end, S2 draws the same tool along the
linen fold in one continuous straight stroke while S1 keeps the book steady. He then
looks toward S1 and says in Mandarin:
<d>[Chinese] 这条书脊已经压平了。</d>
S1 listens silently with closed lips. The camera makes a very slow, smooth push-in while
keeping both pairs of hands, the tool, and the book spine visible. The paper stacks,
thread spools, and wooden press in the background maintain their established layout.
""",
            soundscape=(
                "Warm workshop room tone, linen rubbing against paper, the bone folder "
                "sliding along the book spine, and light apron and sleeve movement."
            ),
        ),
    ),
    ReviewCase(
        case_id="h3_structured_compare_720p15_signal_token_lever",
        resolution="720p",
        duration_seconds=15,
        seed=82715,
        candidates=("c01", "c02", "c03"),
        prompt=_compiled_h3_prompt(
            shot_body="""
A realistic cinematic continuous medium shot inside a restored railway signal cabin at
dusk. The same female signal operator (S1), wearing a dark green uniform, stands beside
a waist-high mechanical console. The same male station clerk (S2), wearing a gray vest,
stands opposite her. A single square brass token rests in S1's right hand. The console
has one matching square slot, one red lever, and one unlit amber indicator.

During the opening five seconds, S1 shows the token to S2 and says clearly in Mandarin:
<d>[Chinese] 区间已经清空，可以放行。</d>
S2 listens silently. From five to ten seconds, S1 extends the token across the console.
S2 first grips its outer edges with his right hand; S1 releases only after their hands
make secure contact. From ten seconds to the end, S2 inserts the same token completely
into the square slot, then pulls the red lever through one smooth arc until it locks.
Only after the lever locks does the amber indicator illuminate. S2 checks the light and
answers in Mandarin:
<d>[Chinese] 信号确认，列车可以通过。</d>
S1 remains silent while watching the console. The camera performs a restrained lateral
move that keeps the token handoff, slot, lever, indicator, and both faces visible. The
window frames, wall clock, and two unused black levers retain their established forms
and positions throughout the take.
""",
            soundscape=(
                "Low signal-cabin ambience, distant rail noise, a light brass-token contact, "
                "the token entering its slot, one mechanical lever clank, and one relay click."
            ),
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--acceleration", type=float, default=75.0)
    parser.add_argument(
        "--batch",
        choices=("original-r01", "h3-structured-r02"),
        default="original-r01",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _progress(run_id: str):
    last: tuple[Any, Any] = (None, None)

    def callback(payload: dict[str, Any]) -> None:
        nonlocal last
        percent = payload.get("percent")
        stage = payload.get("stage")
        bucket = None if percent is None else int(float(percent) // 10) * 10
        marker = (stage, bucket)
        if marker == last:
            return
        last = marker
        print(json.dumps({
            "event": "progress",
            "run_id": run_id,
            "stage": stage,
            "percent": percent,
            "detail": payload.get("detail"),
        }, ensure_ascii=False), flush=True)

    return callback


async def _main() -> int:
    args = _parse_args()
    release_root = args.release_root.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    paths = ServicePaths.defaults(release_root)
    factory = NativeSessionFactory(NativeSessionPaths(
        model_root=paths.model_dir,
        minimax_source=paths.minimax_source_dir,
        lightx_source=paths.lightx_source_dir,
        turbo_curve=paths.turbo_curve_path,
        output_root=output_root,
    ))
    engine = NativeHotH3Engine(factory, output_root=output_root)
    cases = (
        (*STRUCTURED_SHARED_CASES, *STRUCTURED_COMPARISON_CASES)
        if args.batch == "h3-structured-r02"
        else (*SHARED_CASES, DISCRIMINATIVE_CASE)
    )
    expected_runs = sum(len(case.candidates) for case in cases)
    report: dict[str, Any] = {
        "schema_version": "v24_multiscale_human_review_v2",
        "batch": args.batch,
        "purpose": (
            "official H3 structured prompt adherence and three multi-duration C01/C02/C03 comparisons"
            if args.batch == "h3-structured-r02"
            else "cross-duration quality/generalisation review plus 720p12 candidate comparison"
        ),
        "started_at_unix": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "public_controls": {
            "sampling_steps": args.steps,
            "acceleration": args.acceleration,
        },
        "candidate_protocol": {
            "shared_cases": "candidate-independent region; generated once with C03",
            "comparison_cases": "same prompt, seed, geometry and public controls inside each candidate group",
            "prompt_protocol": (
                "compiled integrated_multimodal_description / <d>[Language]> / overall_soundscape / non_diegetic_music"
                if args.batch == "h3-structured-r02"
                else "legacy free-form research prompt"
            ),
            "candidate_selection_does_not_use_prompt_semantics": True,
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_HOME",
                "H3_NATIVE_PARETO_V24",
                "H3_NATIVE_ENABLE_SPARSE",
                "H3_NATIVE_SPARGE_BUILD_DIR",
                "H3_SERVE_RUNTIME_DIR",
                "H3_SERVE_MODEL_DIR",
                "H3_SERVE_MINIMAX_SOURCE",
                "H3_SERVE_LIGHTX_SOURCE",
            )
        },
        "expected_runs": expected_runs,
        "runs": [],
    }
    _write_report(report_path, report)
    failures = 0
    try:
        preload_started = time.monotonic()
        print(json.dumps({"event": "preload_start", "family": "first_last"}), flush=True)
        await engine.preload("first_last")
        report["preload_seconds"] = round(time.monotonic() - preload_started, 3)
        report["warm_state"] = engine.warm_state
        _write_report(report_path, report)
        if engine.warm_state.get("status") != "ready":
            raise RuntimeError(f"failed to preload first_last: {engine.warm_state}")
        if engine._built is None:  # Deliberately fail closed in this research batch.
            raise RuntimeError("hot session missing after successful preload")

        for case in cases:
            for candidate in case.candidates:
                selector = V24ResearchParetoRuntimeSelector(
                    candidate_id=RESEARCH_CALIBRATION_IDS[candidate]
                )
                # One process and one model residency are retained. This
                # direct injection exists only in this offline research tool.
                engine._built.session.v19_selector = selector
                resolved_candidate = selector.candidate.candidate_id
                run_id = f"{case.case_id}_{candidate}"
                spec = GenerationSpec.from_mapping({
                    "prompt": case.prompt,
                    "engine": "original",
                    "resolution": case.resolution,
                    "aspect_ratio": "16:9",
                    "duration_seconds": case.duration_seconds,
                    "sampling_steps": args.steps,
                    "acceleration": args.acceleration,
                    "seed": case.seed,
                })
                output_path = output_root / (
                    f"v24_multiscale_{run_id}_{spec.width}x{spec.height}_"
                    f"{spec.frames}f_seed{case.seed}.mp4"
                )
                run_report: dict[str, Any] = {
                    "run_id": run_id,
                    "status": "running",
                    "candidate_alias": candidate,
                    "candidate_id": resolved_candidate,
                    "shared_surface_case": len(case.candidates) == 1,
                    "spec": spec.to_dict(include_execution=True),
                    "prompt": case.prompt,
                    "output_file": output_path.name,
                    "started_at_unix": time.time(),
                }
                report["runs"].append(run_report)
                _write_report(report_path, report)
                wall_started = time.monotonic()
                print(json.dumps({
                    "event": "run_start",
                    "run_id": run_id,
                    "candidate_id": resolved_candidate,
                    "geometry": [spec.width, spec.height, spec.frames],
                    "actual_duration_seconds": spec.actual_duration_seconds,
                }, ensure_ascii=False), flush=True)
                try:
                    result = await engine.generate(
                        spec=spec,
                        first_frame=None,
                        last_frame=None,
                        reference_images=(),
                        reference_videos=(),
                        reference_audios=(),
                        cancel_event=asyncio.Event(),
                        output_path=output_path,
                        progress_callback=_progress(run_id),
                    )
                    run_report.update({
                        "status": "succeeded",
                        "wall_seconds": round(time.monotonic() - wall_started, 3),
                        "generation_elapsed_seconds": result.elapsed_seconds,
                        "runtime_key": result.runtime_key,
                        "stage_seconds": result.stage_seconds,
                        "inference_plan": result.inference_plan,
                        "output_bytes": result.output_path.stat().st_size,
                        "output_sha256": _sha256(result.output_path),
                    })
                except Exception as error:
                    failures += 1
                    run_report.update({
                        "status": "failed",
                        "wall_seconds": round(time.monotonic() - wall_started, 3),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
                    print(json.dumps({
                        "event": "run_failed",
                        "run_id": run_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }, ensure_ascii=False), flush=True)
                finally:
                    run_report["completed_at_unix"] = time.time()
                    _write_report(report_path, report)
                print(json.dumps({
                    "event": "run_end",
                    "run_id": run_id,
                    "status": run_report["status"],
                    "wall_seconds": run_report.get("wall_seconds"),
                    "stage_seconds": run_report.get("stage_seconds"),
                }, ensure_ascii=False), flush=True)
    finally:
        close_started = time.monotonic()
        await engine.close()
        report["close_seconds"] = round(time.monotonic() - close_started, 3)
        report["completed_at_unix"] = time.time()
        all_succeeded = (
            len(report["runs"]) == expected_runs
            and all(run.get("status") == "succeeded" for run in report["runs"])
        )
        report["status"] = "succeeded" if failures == 0 and all_succeeded else "failed"
        _write_report(report_path, report)
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
