#!/usr/bin/env python3
"""Compare every tensor in one or more formal H3 sampler checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--candidate", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def compare_tree(
    reference: Any,
    candidate: Any,
    *,
    path: str = "$",
) -> tuple[int, list[dict[str, Any]]]:
    if isinstance(reference, torch.Tensor):
        if not isinstance(candidate, torch.Tensor):
            return 1, [{
                "path": path,
                "reason": "candidate_is_not_tensor",
                "affects_tensor_exactness": True,
            }]
        mismatch: dict[str, Any] = {
            "path": path,
            "affects_tensor_exactness": True,
        }
        if reference.dtype != candidate.dtype:
            mismatch.update({
                "reason": "dtype",
                "reference": str(reference.dtype),
                "candidate": str(candidate.dtype),
            })
        elif reference.shape != candidate.shape:
            mismatch.update({
                "reason": "shape",
                "reference": list(reference.shape),
                "candidate": list(candidate.shape),
            })
        elif not torch.equal(reference, candidate):
            mismatch.update({"reason": "tensor_bytes_or_values"})
        else:
            return 1, []
        return 1, [mismatch]

    if isinstance(reference, dict):
        if not isinstance(candidate, dict):
            return 0, [{
                "path": path,
                "reason": "candidate_is_not_dict",
                "affects_tensor_exactness": True,
            }]
        reference_keys = set(reference)
        candidate_keys = set(candidate)
        mismatches: list[dict[str, Any]] = []
        if reference_keys != candidate_keys:
            mismatches.append({
                "path": path,
                "reason": "dict_keys",
                "missing": sorted(reference_keys - candidate_keys),
                "extra": sorted(candidate_keys - reference_keys),
                "affects_tensor_exactness": True,
            })
        count = 0
        for key in sorted(reference_keys & candidate_keys, key=str):
            child_count, child_mismatches = compare_tree(
                reference[key], candidate[key], path=f"{path}.{key}"
            )
            count += child_count
            mismatches.extend(child_mismatches)
        return count, mismatches

    if isinstance(reference, (list, tuple)):
        if not isinstance(candidate, type(reference)):
            return 0, [{
                "path": path,
                "reason": "sequence_type",
                "affects_tensor_exactness": True,
            }]
        if len(reference) != len(candidate):
            return 0, [{
                "path": path,
                "reason": "sequence_length",
                "reference": len(reference),
                "candidate": len(candidate),
                "affects_tensor_exactness": True,
            }]
        count = 0
        mismatches: list[dict[str, Any]] = []
        for index, (reference_item, candidate_item) in enumerate(
            zip(reference, candidate)
        ):
            child_count, child_mismatches = compare_tree(
                reference_item,
                candidate_item,
                path=f"{path}[{index}]",
            )
            count += child_count
            mismatches.extend(child_mismatches)
        return count, mismatches

    if reference != candidate:
        return 0, [{
            "path": path,
            "reason": "value",
            "reference": repr(reference),
            "candidate": repr(candidate),
            "affects_tensor_exactness": False,
        }]
    return 0, []


def main() -> int:
    args = parse_args()
    reference_path = args.reference.resolve()
    reference = torch.load(
        reference_path, map_location="cpu", weights_only=True
    )
    comparisons = []
    all_tensors_exact = True
    all_metadata_exact = True
    for candidate_arg in args.candidate:
        candidate_path = candidate_arg.resolve()
        candidate = torch.load(
            candidate_path, map_location="cpu", weights_only=True
        )
        tensor_count, mismatches = compare_tree(reference, candidate)
        tensor_exact = not any(
            mismatch["affects_tensor_exactness"]
            for mismatch in mismatches
        )
        metadata_exact = not mismatches
        all_tensors_exact = all_tensors_exact and tensor_exact
        all_metadata_exact = all_metadata_exact and metadata_exact
        comparisons.append({
            "candidate": str(candidate_path),
            "tensor_count": tensor_count,
            "tensor_exact": tensor_exact,
            "metadata_exact": metadata_exact,
            "mismatches": mismatches,
        })
        del candidate
    report = {
        "schema_version": "h3_formal_checkpoint_exact_comparison_v1",
        "reference": str(reference_path),
        "all_tensors_exact": all_tensors_exact,
        "all_metadata_exact": all_metadata_exact,
        "comparisons": comparisons,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if all_tensors_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
