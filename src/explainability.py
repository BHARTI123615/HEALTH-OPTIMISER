"""Objective-based explanations for synthetic optimizer choices."""


def explain_selection(selected: list[dict], risk_weight: float = 1.0,
                      cost_weight: float = 0.5) -> list[dict]:
    """Explain each selected option's contribution to the configured objective.

    These explanations describe the optimizer objective, not the ML model or a
    clinical rationale.
    """
    explanations = []
    for candidate in selected:
        contribution = (
            candidate["benefit"]
            - risk_weight * candidate["risk"]
            - cost_weight * candidate["cost"]
        )
        explanations.append({
            "Candidate": candidate["name"],
            "Illustrative benefit": candidate["benefit"],
            "Illustrative risk": candidate["risk"],
            "Illustrative cost": candidate["cost"],
            "Objective contribution": round(contribution, 2),
            "Explanation": "Selected by the optimization objective and subject to the configured constraints.",
        })
    return explanations


def explain(solution: dict, validation: dict) -> str:
    """Return a short result summary for the dashboard."""
    names = [item["name"] for item in solution.get("selected", [])]
    choices = ", ".join(names) if names else "no options (no feasible selection)"
    state = "passed" if validation.get("valid") else "failed"
    return (f"The {solution.get('solver', 'optimizer')} selected {choices}. "
            f"Illustrative objective: {solution.get('objective', 0):.2f}; "
            f"resource units used: {solution.get('resource_used', 0)}. Independent feasibility checks {state}. "
            "This is an educational optimization result, not a treatment recommendation.")
