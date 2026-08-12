"""Validation and agreement gates for independently reviewed Scene materials."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any


def validate(record: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator, FormatChecker

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = [
        f"{'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(record), key=lambda item: list(item.path))
    ]
    reviews = record.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        return errors
    reviewers = [str(review.get("reviewer", "")).strip().casefold() for review in reviews]
    if reviewers[0] and reviewers[0] == reviewers[1]:
        errors.append("reviews: independent reviewers must be distinct")
    for index, review in enumerate(reviews):
        value = str(review.get("reviewed_at", ""))
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"reviews.{index}.reviewed_at: invalid ISO-8601 timestamp")
    adjudication = record.get("adjudication") or {}
    left = reviews[0].get("material_class")
    right = reviews[1].get("material_class")
    status = adjudication.get("status")
    resolved = adjudication.get("material_class")
    if left == right:
        if status != "agreed":
            errors.append("adjudication.status: matching reviews require 'agreed'")
        if resolved != left:
            errors.append("adjudication.material_class: must equal the agreed reviews")
    elif status == "adjudicated":
        if not resolved or not adjudication.get("adjudicator") or not adjudication.get("reason"):
            errors.append("adjudication: disagreements require class, adjudicator, and reason")
    elif status != "unresolvable":
        errors.append("adjudication.status: disagreement must be adjudicated or unresolvable")
    return errors


def outcome(record: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    adjudication = record.get("adjudication") or {}
    status = adjudication.get("status")
    eligible = not errors and status in {"agreed", "adjudicated"}
    return {
        "asset_id": record.get("asset_id"),
        "surface_id": record.get("surface_id"),
        "valid": not errors,
        "errors": errors,
        "review_agreement": (
            len(record.get("reviews") or []) == 2
            and record["reviews"][0].get("material_class")
            == record["reviews"][1].get("material_class")
        ),
        "adjudication_status": status,
        "material_class": adjudication.get("material_class") if eligible else None,
        "label_eligible": eligible,
    }


def agreement(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (record["reviews"][0]["material_class"], record["reviews"][1]["material_class"])
        for record in records
        if isinstance(record.get("reviews"), list) and len(record["reviews"]) == 2
    ]
    if not pairs:
        return {"scored": 0, "agreements": 0, "raw_agreement": None, "cohen_kappa": None}
    agreements = sum(left == right for left, right in pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    labels = set(left_counts) | set(right_counts)
    total = len(pairs)
    observed = agreements / total
    expected = sum(
        (left_counts[label] / total) * (right_counts[label] / total)
        for label in labels
    )
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else None
    return {
        "scored": total,
        "agreements": agreements,
        "raw_agreement": observed,
        "cohen_kappa": kappa,
        "expected_agreement": expected,
    }
