"""Binary candidate optimization with a classical baseline and Qiskit QUBO path."""
from __future__ import annotations

from itertools import product
from time import perf_counter


def _score(candidate: dict, risk_weight: float, cost_weight: float) -> float:
    return candidate["benefit"] - risk_weight * candidate["risk"] - cost_weight * candidate["cost"]


def solve_classically(candidates: list[dict], risk_weight: float = 1.0,
                      cost_weight: float = 0.5, max_resource: int = 2) -> dict:
    """Exhaustively find the best feasible binary selection for the small demo."""
    if max_resource < 0:
        raise ValueError("max_resource must be non-negative.")
    best_score, best_selected = 0.0, []
    for bits in product((0, 1), repeat=len(candidates)):
        selected = [item for item, bit in zip(candidates, bits) if bit]
        if sum(item["resource"] for item in selected) > max_resource:
            continue
        score = sum(_score(item, risk_weight, cost_weight) for item in selected)
        if score > best_score:
            best_score, best_selected = score, selected
    return {"selected": best_selected, "objective": float(best_score),
            "resource_used": sum(item["resource"] for item in best_selected),
            "solver": "classical exact enumeration"}


def build_quadratic_program(candidates: list[dict], risk_weight: float = 1.0,
                            cost_weight: float = 0.5, max_resource: int = 2):
    """Build the constrained binary Qiskit model for the synthetic objective."""
    from qiskit_optimization import QuadraticProgram

    qp = QuadraticProgram("QuantumHealth_Demo")
    linear = {}
    resource = {}
    for candidate in candidates:
        variable = f"x_{candidate['id']}"
        qp.binary_var(name=variable)
        linear[variable] = _score(candidate, risk_weight, cost_weight)
        resource[variable] = candidate["resource"]
    qp.maximize(linear=linear)
    qp.linear_constraint(linear=resource, sense="LE", rhs=max_resource, name="resource_budget")
    return qp


def build_qubo(candidates: list[dict], risk_weight: float = 1.0,
               cost_weight: float = 0.5, max_resource: int = 2):
    """Convert the constrained quadratic program into a QUBO with Qiskit."""
    from qiskit_optimization.converters import QuadraticProgramToQubo

    return QuadraticProgramToQubo().convert(
        build_quadratic_program(candidates, risk_weight, cost_weight, max_resource)
    )


def solve(candidates: list[dict], max_resource: int = 2, prefer_quantum: bool = False,
          risk_weight: float = 1.0, cost_weight: float = 0.5,
          qaoa_seed: int = 42, qaoa_maxiter: int = 30) -> dict:
    """Compare exact classical optimization with a reproducible local QAOA run."""
    classical_started = perf_counter()
    baseline = solve_classically(candidates, risk_weight, cost_weight, max_resource)
    classical_seconds = perf_counter() - classical_started
    classical_summary = {
        "objective": baseline["objective"],
        "selected_ids": [item["id"] for item in baseline["selected"]],
        "resource_used": baseline["resource_used"],
        "execution_seconds": classical_seconds,
    }
    if not prefer_quantum:
        return {**baseline, "quantum_status": "not requested",
                "classical_comparison": classical_summary}

    try:
        import numpy as np
        from qiskit.exceptions import QiskitError
        from qiskit.primitives import StatevectorSampler
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
    except (ImportError, ModuleNotFoundError) as exc:
        return _quantum_fallback(baseline, classical_summary, exc, 0.0, qaoa_seed, qaoa_maxiter)

    quantum_started = perf_counter()
    try:
        qubo = build_qubo(candidates, risk_weight, cost_weight, max_resource)
        rng = np.random.default_rng(qaoa_seed)
        initial_point = rng.uniform(-np.pi, np.pi, size=2)
        result = MinimumEigenOptimizer(
            QAOA(
                sampler=StatevectorSampler(seed=qaoa_seed),
                optimizer=COBYLA(maxiter=qaoa_maxiter),
                reps=1,
                initial_point=initial_point,
            )
        ).solve(qubo)
        selected = [
            item for item in candidates
            if result.x[qubo.variables_index[f"x_{item['id']}"]] > 0.5
        ]
        objective = sum(_score(item, risk_weight, cost_weight) for item in selected)
        resource_used = sum(item["resource"] for item in selected)
        quantum_seconds = perf_counter() - quantum_started
        feasible = resource_used <= max_resource
        quantum_summary = {
            "status": "completed" if feasible else "completed; infeasible candidate rejected",
            "solver": "QAOA with local StatevectorSampler",
            "selected_ids": [item["id"] for item in selected],
            "candidate_bitstring": "".join(
                "1" if any(item["id"] == candidate["id"] for item in selected) else "0"
                for candidate in candidates
            ),
            "objective": float(objective),
            "resource_used": resource_used,
            "execution_seconds": quantum_seconds,
            "configuration": {
                "backend": "local statevector simulator",
                "qaoa_reps": 1,
                "classical_optimizer": "COBYLA",
                "max_iterations": qaoa_maxiter,
                "seed": qaoa_seed,
            },
        }
        if not feasible:
            return {
                **baseline,
                "quantum_status": "completed; infeasible result rejected, classical baseline shown",
                "classical_comparison": classical_summary,
                "quantum_result": quantum_summary,
            }
        return {
            "selected": selected,
            "objective": float(objective),
            "resource_used": resource_used,
            "solver": "Qiskit QAOA (local StatevectorSampler)",
            "quantum_status": "completed",
            "classical_comparison": classical_summary,
            "quantum_result": quantum_summary,
        }
    except (ValueError, RuntimeError, TypeError, QiskitError) as exc:
        quantum_seconds = perf_counter() - quantum_started
        return _quantum_fallback(
            baseline, classical_summary, exc, quantum_seconds, qaoa_seed, qaoa_maxiter
        )


def _quantum_fallback(
    baseline: dict,
    classical_summary: dict,
    error: Exception,
    execution_seconds: float,
    seed: int,
    max_iterations: int,
) -> dict:
    """Keep the exact baseline available while explicitly reporting QAOA failure."""
    detail = f"{type(error).__name__}: {error}"
    return {
        **baseline,
        "quantum_status": f"failed; classical fallback used: {detail}",
        "classical_comparison": classical_summary,
        "quantum_result": {
            "status": "failed",
            "error": detail,
            "execution_seconds": execution_seconds,
            "configuration": {
                "backend": "local statevector simulator",
                "qaoa_reps": 1,
                "classical_optimizer": "COBYLA",
                "max_iterations": max_iterations,
                "seed": seed,
            },
        },
    }
