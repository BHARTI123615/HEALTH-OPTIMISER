"""Independent checks for candidate eligibility and resource feasibility."""


def validate_selection(selected: list[dict], eligible_candidates: list[dict], max_resource: int = 2) -> dict:
    """Check eligibility, duplicates, and resource use against original constraints."""
    eligible_ids = {candidate["id"] for candidate in eligible_candidates}
    selected_ids = [candidate["id"] for candidate in selected]
    checks = {
        "all_candidates_eligible": all(candidate_id in eligible_ids for candidate_id in selected_ids),
        "no_duplicate_candidates": len(selected_ids) == len(set(selected_ids)),
        "resource_budget_satisfied": sum(item["resource"] for item in selected) <= max_resource,
    }
    checks["all_checks_passed"] = all(checks.values())
    return checks


def validate_solution(solution: dict, patient: dict, candidates: list[dict], max_resource: int = 2) -> dict:
    """Compatibility wrapper returning a concise valid flag plus detailed checks."""
    checks = validate_selection(solution.get("selected", []), candidates, max_resource)
    return {"valid": checks["all_checks_passed"], "checks": checks}
