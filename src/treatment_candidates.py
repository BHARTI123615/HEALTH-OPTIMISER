"""Fictional intervention options and explicit demo exclusions."""
from __future__ import annotations

CANDIDATES = [
    {"id": "A", "name": "Lifestyle Support", "benefit": 7.0, "risk": 1.0, "cost": 2.0, "resource": 1},
    {"id": "B", "name": "Monitoring Program", "benefit": 5.0, "risk": 0.5, "cost": 1.0, "resource": 1},
    {"id": "C", "name": "Specialist Review", "benefit": 8.0, "risk": 2.0, "cost": 4.0, "resource": 2},
    {"id": "D", "name": "Follow-up Support", "benefit": 4.0, "risk": 0.5, "cost": 1.5, "resource": 1},
]


def generate_candidates() -> list[dict]:
    """Return independent copies of the synthetic candidate definitions."""
    return [candidate.copy() for candidate in CANDIDATES]


def get_eligible_candidates(candidates: list[dict], excluded_ids: list[str] | set[str]) -> list[dict]:
    """Apply explicit user-selected demo exclusions before optimization."""
    excluded = set(excluded_ids)
    unknown = excluded - {candidate["id"] for candidate in candidates}
    if unknown:
        raise ValueError(f"Unknown candidate IDs: {sorted(unknown)}")
    return [candidate.copy() for candidate in candidates if candidate["id"] not in excluded]
