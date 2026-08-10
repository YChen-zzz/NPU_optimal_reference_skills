#!/usr/bin/env python3
"""Fail-closed gate for Phase 3 wave and action execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


TEMPLATE = {
    "schema_version": 1,
    "wave_id": "wave-001",
    "npu_l1": {
        "status": "valid",
        "profile_refs": ["evidence_db/artifacts/npu_l1"],
        "regime_coverage": ["regime-small", "regime-medium", "regime-large"],
    },
    "fast_run": {
        "status": "ready",
        "script_ref": "benchmarks/short_run.py",
        "baseline_result_ref": "evidence_db/short_runs/baseline.yaml",
        "target_seconds": 60,
    },
    "candidates_ref": "evidence_db/candidates.csv",
    "supernodes": [
        {
            "supernode_id": "sn-001",
            "priority": "high",
            "lab_ref": "benchmarks/supernodes/sn-001.py",
            "lab_not_required": "",
        }
    ],
    "actions": [
        {
            "candidate_id": "cand-api",
            "supernode_id": "sn-001",
            "implementation_path": "official_npu_api",
            "status": "proposed",
            "lab_gate": "passed",
            "lab_result_ref": "evidence_db/supernode_labs/sn-001/api.yaml",
            "validation_gate": "weighted_short_run",
            "rollback_plan": "independent commit or switch",
        },
        {
            "candidate_id": "cand-compile",
            "supernode_id": "sn-001",
            "implementation_path": "selective_compile",
            "status": "proposed",
            "lab_gate": "passed",
            "lab_result_ref": "evidence_db/supernode_labs/sn-001/result.yaml",
            "validation_gate": "weighted_short_run",
            "rollback_plan": "independent commit or switch",
            "api_first": {
                "status": "insufficient_gain",
                "evidence_refs": ["evidence_db/supernode_labs/sn-001/api.yaml"],
            },
            "compile_unlock": {
                "other_non_compile_paths": "insufficient_gain",
                "non_compile_backlog_exhausted": True,
                "evidence_refs": ["evidence_db/supernode_labs/sn-001/result.yaml"],
            },
        },
    ],
}

UNLOCKED = {"not_applicable", "tested_rejected", "insufficient_gain"}
IMPLEMENTATION_PATHS = {
    "remove_or_cache",
    "official_npu_api",
    "manual_rewrite",
    "schedule_or_autograd",
    "custom_kernel",
    "selective_compile",
}


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _resolve(workspace: Path, value: object) -> Path | None:
    if not _text(value):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else workspace / path


def _require_ref(errors: list[str], workspace: Path, value: object, label: str) -> None:
    path = _resolve(workspace, value)
    if path is None:
        errors.append(f"{label}: missing path")
    elif not path.exists():
        errors.append(f"{label}: path does not exist: {path}")


def _validate_wave(data: dict, workspace: Path) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not _text(data.get("wave_id")):
        errors.append("wave_id is required")

    npu_l1 = data.get("npu_l1") or {}
    if npu_l1.get("status") != "valid":
        errors.append("npu_l1.status must be valid")
    refs = npu_l1.get("profile_refs") or []
    if not refs:
        errors.append("npu_l1.profile_refs is required")
    for index, ref in enumerate(refs):
        _require_ref(errors, workspace, ref, f"npu_l1.profile_refs[{index}]")
    if not npu_l1.get("regime_coverage"):
        errors.append("npu_l1.regime_coverage is required")

    fast_run = data.get("fast_run") or {}
    if fast_run.get("status") != "ready":
        errors.append("fast_run.status must be ready")
    _require_ref(errors, workspace, fast_run.get("script_ref"), "fast_run.script_ref")
    _require_ref(
        errors,
        workspace,
        fast_run.get("baseline_result_ref"),
        "fast_run.baseline_result_ref",
    )
    target = fast_run.get("target_seconds")
    if not isinstance(target, (int, float)) or target <= 0:
        errors.append("fast_run.target_seconds must be positive")

    _require_ref(errors, workspace, data.get("candidates_ref"), "candidates_ref")

    supernodes = data.get("supernodes") or []
    if not supernodes:
        errors.append("supernodes must not be empty")
    for index, node in enumerate(supernodes):
        prefix = f"supernodes[{index}]"
        if not isinstance(node, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if not _text(node.get("supernode_id")):
            errors.append(f"{prefix}.supernode_id is required")
        if node.get("priority") == "high":
            lab_ref = node.get("lab_ref")
            reason = node.get("lab_not_required")
            if _text(lab_ref):
                _require_ref(errors, workspace, lab_ref, f"{prefix}.lab_ref")
            elif not _text(reason) or len(str(reason).strip()) < 12:
                errors.append(
                    f"{prefix}: high-priority node needs an existing lab_ref "
                    "or a specific lab_not_required reason"
                )

    actions = data.get("actions") or []
    if not actions:
        errors.append("actions must not be empty")
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            errors.append(f"actions[{index}] must be an object")
    valid_actions = [action for action in actions if isinstance(action, dict)]
    action_ids = [action.get("candidate_id") for action in valid_actions]
    if any(not _text(action_id) for action_id in action_ids):
        errors.append("every action needs candidate_id")
    if len(action_ids) != len(set(action_ids)):
        errors.append("action candidate_id values must be unique")

    candidates_path = _resolve(workspace, data.get("candidates_ref"))
    if candidates_path is not None and candidates_path.exists():
        try:
            with candidates_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "candidate_id" not in reader.fieldnames:
                    errors.append("candidates_ref must contain a candidate_id column")
                else:
                    known = {row.get("candidate_id") for row in reader}
                    missing = [action_id for action_id in action_ids if action_id not in known]
                    if missing:
                        errors.append(
                            "actions missing from candidates_ref: " + ", ".join(missing)
                        )
        except OSError as exc:
            errors.append(f"cannot read candidates_ref: {exc}")
    return errors


def _find_action(data: dict, candidate_id: str) -> dict | None:
    matches = [
        action
        for action in data.get("actions") or []
        if action.get("candidate_id") == candidate_id
    ]
    return matches[0] if len(matches) == 1 else None


def _validate_action(data: dict, workspace: Path, candidate_id: str) -> list[str]:
    errors = _validate_wave(data, workspace)
    action = _find_action(data, candidate_id)
    if action is None:
        errors.append(f"candidate_id must identify exactly one action: {candidate_id}")
        return errors

    if action.get("lab_gate") not in {"passed", "not_required"}:
        errors.append("action.lab_gate must be passed or not_required")
    elif action.get("lab_gate") == "passed":
        _require_ref(
            errors,
            workspace,
            action.get("lab_result_ref"),
            "action.lab_result_ref",
        )
    elif not _text(action.get("lab_not_required")):
        errors.append("action.lab_not_required needs a specific reason")
    if not _text(action.get("validation_gate")):
        errors.append("action.validation_gate is required")
    if not _text(action.get("rollback_plan")):
        errors.append("action.rollback_plan is required")

    path = action.get("implementation_path")
    if path not in IMPLEMENTATION_PATHS:
        errors.append(f"unknown action.implementation_path: {path}")
    if path not in {"remove_or_cache", "official_npu_api"}:
        api_first = action.get("api_first") or {}
        if api_first.get("status") not in UNLOCKED:
            errors.append(
                "API-first gate: status must be not_applicable, tested_rejected, "
                "or insufficient_gain"
            )
        api_evidence = api_first.get("evidence_refs") or []
        if not api_evidence:
            errors.append("API-first gate: evidence_refs is required")
        for index, ref in enumerate(api_evidence):
            _require_ref(errors, workspace, ref, f"api_first.evidence_refs[{index}]")

    if path == "selective_compile":
        unlock = action.get("compile_unlock") or {}
        if unlock.get("other_non_compile_paths") not in UNLOCKED:
            errors.append(
                "compile locked: other_non_compile_paths must be not_applicable, "
                "tested_rejected, or insufficient_gain"
            )
        if unlock.get("non_compile_backlog_exhausted") is not True:
            errors.append("compile locked: non_compile_backlog_exhausted must be true")
        evidence = unlock.get("evidence_refs") or []
        if not evidence:
            errors.append("compile locked: compile_unlock.evidence_refs is required")
        for index, ref in enumerate(evidence):
            _require_ref(
                errors,
                workspace,
                ref,
                f"compile_unlock.evidence_refs[{index}]",
            )
    return errors


def _receipt_path(
    workspace: Path,
    data: dict,
    stage: str,
    candidate_id: str | None,
) -> Path:
    raw_name = f"{data['wave_id']}-{stage}-{candidate_id or 'all'}"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name)
    receipt_dir = workspace / "evidence_db" / "preflight_receipts"
    return receipt_dir / f"{safe_name}.json"


def _write_receipt(
    workspace: Path,
    manifest: Path,
    data: dict,
    stage: str,
    candidate_id: str | None,
) -> Path:
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipt = _receipt_path(workspace, data, stage, candidate_id)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "stage": stage,
        "wave_id": data["wave_id"],
        "candidate_id": candidate_id,
        "manifest": str(manifest),
        "manifest_sha256": digest,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def _verify_receipt(
    workspace: Path,
    manifest: Path,
    data: dict,
    stage: str,
    candidate_id: str | None,
) -> tuple[Path, list[str]]:
    receipt = _receipt_path(workspace, data, stage, candidate_id)
    errors: list[str] = []
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return receipt, [f"cannot read receipt: {exc}"]
    current_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if payload.get("status") != "pass":
        errors.append("receipt status is not pass")
    if payload.get("manifest_sha256") != current_hash:
        errors.append("receipt manifest hash is stale")
    if payload.get("candidate_id") != candidate_id:
        errors.append("receipt candidate_id does not match")
    return receipt, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evidence_db/phase3_preflight.json"),
    )
    parser.add_argument("--stage", choices=("wave", "action"), default="wave")
    parser.add_argument("--candidate-id")
    parser.add_argument("--print-template", action="store_true")
    parser.add_argument("--verify-receipt", action="store_true")
    options = parser.parse_args()

    if options.print_template:
        print(json.dumps(TEMPLATE, ensure_ascii=False, indent=2))
        return 0
    if options.stage == "action" and not options.candidate_id:
        parser.error("--candidate-id is required for --stage action")

    workspace = options.workspace.resolve()
    manifest = options.manifest
    if not manifest.is_absolute():
        manifest = workspace / manifest
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: cannot read preflight manifest: {exc}", file=sys.stderr)
        return 2
    if not isinstance(data, dict):
        print("BLOCKED: preflight manifest root must be an object", file=sys.stderr)
        return 2

    errors = (
        _validate_wave(data, workspace)
        if options.stage == "wave"
        else _validate_action(data, workspace, options.candidate_id)
    )
    if errors:
        print("BLOCKED: Phase 3 preflight failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    if options.verify_receipt:
        receipt, receipt_errors = _verify_receipt(
            workspace,
            manifest,
            data,
            options.stage,
            options.candidate_id,
        )
        if receipt_errors:
            print("BLOCKED: Phase 3 receipt verification failed", file=sys.stderr)
            for error in receipt_errors:
                print(f"- {error}", file=sys.stderr)
            return 2
        print(f"PASS: receipt matches current manifest: {receipt}")
        return 0

    receipt = _write_receipt(
        workspace,
        manifest,
        data,
        options.stage,
        options.candidate_id,
    )
    suffix = "wave" if options.stage == "wave" else f"action {options.candidate_id}"
    print(f"PASS: Phase 3 {suffix} preflight")
    print(f"receipt: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
