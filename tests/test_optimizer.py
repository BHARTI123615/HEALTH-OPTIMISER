import pandas as pd
import pytest

from src.data_processing import validate_patient
from src.ml_model import (
    build_demo_dataset,
    evaluate_demo_model,
    train_demo_model,
    predict_demo_probability,
)
from src.treatment_candidates import generate_candidates, get_eligible_candidates
from src.optimizer import build_qubo, solve, solve_classically
from src.validation import validate_selection
from src.explainability import explain_selection


def sample():
    return {"age": 45, "bmi": 24.5, "heart_rate": 72, "systolic_bp": 120, "symptom_score": 2}


def test_profile_and_synthetic_model():
    patient = validate_patient(sample())
    model, accuracy = train_demo_model()
    probability = predict_demo_probability(model, patient)
    metrics = evaluate_demo_model(model)
    assert 0 <= probability <= 1
    assert 0 <= accuracy <= 1
    assert metrics["Accuracy"] == pytest.approx(accuracy)
    assert set(metrics) == {"Accuracy", "Precision", "Recall", "F1"}
    assert all(0 <= value <= 1 for value in metrics.values())


def test_invalid_patient_input():
    patient = sample()
    patient["age"] = 12
    with pytest.raises(ValueError, match="age must be between"):
        validate_patient(patient)


def test_demo_dataset_is_reproducible_and_has_both_labels():
    first, second = build_demo_dataset(), build_demo_dataset()
    assert first.equals(second)
    assert set(first["demo_label"].unique()) == {0, 1}


def test_probability_requires_one_row_and_all_features():
    model, _ = train_demo_model()
    with pytest.raises(ValueError):
        predict_demo_probability(model, pd.DataFrame({"age": [40]}))


def test_candidate_generation_returns_independent_copies():
    first = generate_candidates()
    first[0]["name"] = "mutated"
    assert generate_candidates()[0]["name"] == "Lifestyle Support"


def test_explicit_exclusions_are_applied_before_optimization():
    candidates = generate_candidates()
    eligible = get_eligible_candidates(candidates, ["A", "C"])
    assert [item["id"] for item in eligible] == ["B", "D"]
    assert [item["id"] for item in candidates] == ["A", "B", "C", "D"]
    with pytest.raises(ValueError):
        get_eligible_candidates(candidates, ["unknown"])


def test_classical_optimizer_finds_best_resource_feasible_selection():
    candidates = generate_candidates()
    result = solve_classically(candidates, max_resource=2)
    assert [item["id"] for item in result["selected"]] == ["A", "B"]
    assert result["resource_used"] <= 2
    assert result["objective"] == pytest.approx(9.0)


def test_qubo_conversion_removes_constraints():
    candidates = generate_candidates()
    qubo = build_qubo(candidates)
    assert qubo.get_num_vars() > len(candidates)
    assert len(qubo.linear_constraints) == 0


def test_qaoa_runs_on_local_simulator_and_returns_feasible_result():
    candidates = generate_candidates()
    result = solve(candidates, max_resource=2, prefer_quantum=True, qaoa_maxiter=12)
    quantum = result["quantum_result"]
    assert quantum["status"] == "completed"
    assert len(quantum["candidate_bitstring"]) == len(candidates)
    assert quantum["resource_used"] <= 2
    assert result["quantum_status"] == "completed"
    assert result["classical_comparison"]["objective"] == pytest.approx(9.0)
    assert result["solver"] == "Qiskit QAOA (local StatevectorSampler)"


def test_solution_is_independently_validated():
    candidates = generate_candidates()
    eligible = get_eligible_candidates(candidates, ["A"])
    result = solve(eligible, max_resource=2)
    checks = validate_selection(result["selected"], eligible, max_resource=2)
    assert checks["all_checks_passed"]


def test_validator_detects_excluded_duplicate_and_over_resource_items():
    candidates = generate_candidates()
    checks = validate_selection([candidates[0], candidates[0], candidates[2]],
                                candidates[1:], max_resource=2)
    assert checks["all_candidates_eligible"] is False
    assert checks["no_duplicate_candidates"] is False
    assert checks["resource_budget_satisfied"] is False
    assert checks["all_checks_passed"] is False


def test_selection_explanation_shows_objective_contribution():
    candidate = generate_candidates()[0]
    explanation = explain_selection([candidate], risk_weight=1.0, cost_weight=0.5)
    assert explanation[0]["Candidate"] == "Lifestyle Support"
    assert explanation[0]["Objective contribution"] == pytest.approx(5.0)
    assert "configured constraints" in explanation[0]["Explanation"]
