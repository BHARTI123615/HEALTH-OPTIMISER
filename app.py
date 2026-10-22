"""Streamlit interface for the QuantumHealth Optimizer educational prototype."""
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import validate_patient
from src.explainability import explain, explain_selection
from src.ml_model import evaluate_demo_model, predict_demo_probability, train_demo_model
from src.optimizer import build_qubo, solve
from src.treatment_candidates import generate_candidates, get_eligible_candidates
from src.validation import validate_selection

st.set_page_config(page_title="QuantumHealth Optimizer", page_icon="⚛️", layout="wide")
st.title("QuantumHealth Optimizer")
st.caption("Hybrid quantum-classical optimization experiment")
st.warning(
    "Research and educational prototype only. Uses synthetic data and fictional option "
    "attributes. Not clinically validated; it does not diagnose or recommend medicines or treatment."
)

DEFAULT_INPUTS = {
    "age": 45,
    "bmi": 24.0,
    "heart_rate": 72,
    "systolic_bp": 120,
    "symptom_score": 2.0,
    "excluded_ids": [],
    "max_resource": 2,
    "risk_weight": 1.0,
    "cost_weight": 0.5,
    "quantum": False,
    "quantum_trials": 2,
}
for input_key, input_value in DEFAULT_INPUTS.items():
    st.session_state.setdefault(input_key, input_value)
st.session_state.setdefault("run_history", [])


def reset_demo_inputs() -> None:
    for key, value in DEFAULT_INPUTS.items():
        st.session_state[key] = value
    st.session_state.pop("previous_run", None)


@st.cache_resource
def get_demo_model():
    return train_demo_model()


with st.sidebar:
    st.button("Reset demonstration inputs", on_click=reset_demo_inputs)
    with st.form("patient_profile"):
        st.header("Synthetic profile")
        age = st.number_input("Age", min_value=18, max_value=100, key="age")
        bmi = st.number_input(
            "BMI (synthetic)", min_value=10.0, max_value=80.0, key="bmi"
        )
        heart_rate = st.number_input(
            "Heart rate (synthetic)", min_value=30, max_value=220, key="heart_rate"
        )
        systolic_bp = st.number_input(
            "Systolic blood pressure (synthetic)",
            min_value=50,
            max_value=250,
            key="systolic_bp",
        )
        symptom_score = st.slider(
            "Symptom score (0–10, synthetic)", 0.0, 10.0, key="symptom_score"
        )
        excluded_ids = st.multiselect(
            "Exclude hypothetical options", ["A", "B", "C", "D"], key="excluded_ids"
        )
        max_resource = st.slider(
            "Maximum resource units", 0, 5, key="max_resource"
        )
        risk_weight = st.slider(
            "Illustrative risk weight", 0.0, 3.0, step=0.1, key="risk_weight"
        )
        cost_weight = st.slider(
            "Illustrative cost weight", 0.0, 3.0, step=0.1, key="cost_weight"
        )
        quantum = st.checkbox(
            "Run local QAOA experiment (statevector simulator)", key="quantum"
        )
        quantum_trials = st.slider(
            "Reproducible QAOA trials", 1, 3, key="quantum_trials",
        )
        submitted = st.form_submit_button("Run / Re-optimize", type="primary")

if submitted:
    try:
        profile = {
            "age": age,
            "bmi": bmi,
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "symptom_score": symptom_score,
        }
        patient_frame = validate_patient(profile)
        model, accuracy = get_demo_model()
        probability = predict_demo_probability(model, patient_frame)
        evaluation = evaluate_demo_model(model)
        all_candidates = generate_candidates()
        candidates = get_eligible_candidates(all_candidates, excluded_ids)

        classical = solve(
            candidates,
            max_resource=max_resource,
            risk_weight=risk_weight,
            cost_weight=cost_weight,
        )
        trial_results = []
        if quantum:
            for trial_index in range(quantum_trials):
                trial_results.append(
                    solve(
                        candidates,
                        max_resource=max_resource,
                        prefer_quantum=True,
                        risk_weight=risk_weight,
                        cost_weight=cost_weight,
                        qaoa_seed=42 + trial_index,
                    )
                )

        completed_trials = [
            result for result in trial_results
            if result["quantum_result"]["status"] == "completed"
        ]
        feasible_trials = [
            result for result in completed_trials
            if validate_selection(
                result["quantum_result"]["selected_ids"]
                and [
                    candidate for candidate in candidates
                    if candidate["id"] in result["quantum_result"]["selected_ids"]
                ],
                candidates,
                max_resource,
            )["all_checks_passed"]
        ]
        if feasible_trials:
            solution = max(
                feasible_trials,
                key=lambda item: item["quantum_result"]["objective"],
            )
        else:
            solution = classical
        validation = {
            "valid": validate_selection(
                solution["selected"], candidates, max_resource=max_resource
            )["all_checks_passed"],
            "checks": validate_selection(
                solution["selected"], candidates, max_resource=max_resource
            ),
        }

        left, right = st.columns(2)
        left.metric("Synthetic positive-class probability", f"{probability:.1%}")
        left.caption("Artificial label output; not a disease probability or clinical risk score.")
        right.metric("Resource units used", f"{solution['resource_used']} / {max_resource}")

        st.subheader("Synthetic model evaluation")
        st.caption(
            f"All metrics measure agreement with the artificial label on a fixed synthetic "
            f"holdout set; they are not clinical performance measures. Accuracy: {accuracy:.1%}."
        )
        metrics_frame = pd.DataFrame(
            {"Metric": list(evaluation), "Score": list(evaluation.values())}
        )
        metric_chart = px.bar(
            metrics_frame,
            x="Metric",
            y="Score",
            range_y=[0, 1],
            title="Held-out synthetic-label evaluation",
            text_auto=".0%",
        )
        st.plotly_chart(metric_chart, use_container_width=True)

        st.subheader("Hypothetical option attributes")
        candidate_frame = pd.DataFrame(all_candidates)
        candidate_frame = candidate_frame.rename(
            columns={"name": "Option", "benefit": "Benefit", "risk": "Risk", "cost": "Cost"}
        )
        st.plotly_chart(
            px.bar(
                candidate_frame,
                x="Option",
                y=["Benefit", "Risk", "Cost"],
                barmode="group",
                title="Illustrative candidate values (not clinical measurements)",
            ),
            use_container_width=True,
        )
        excluded = set(excluded_ids)
        st.dataframe(
            pd.DataFrame([
                {
                    "ID": candidate["id"],
                    "Option": candidate["name"],
                    "Benefit": candidate["benefit"],
                    "Risk": candidate["risk"],
                    "Cost": candidate["cost"],
                    "Resource": candidate["resource"],
                    "Eligible": candidate["id"] not in excluded,
                    "Reason": "Excluded by demo selection" if candidate["id"] in excluded else "",
                }
                for candidate in all_candidates
            ]),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Classical and QAOA comparison")
        comparison_rows = [{
            "Method": "Classical exact search",
            "Trial": "Baseline",
            "Objective": classical["objective"],
            "Execution seconds": classical["classical_comparison"]["execution_seconds"],
            "Status": "completed",
        }]
        for index, result in enumerate(trial_results, start=1):
            quantum_result = result["quantum_result"]
            comparison_rows.append({
                "Method": "QAOA local simulator",
                "Trial": f"Trial {index}",
                "Objective": quantum_result.get("objective"),
                "Execution seconds": quantum_result["execution_seconds"],
                "Status": quantum_result["status"],
            })
        comparison_frame = pd.DataFrame(comparison_rows)
        st.plotly_chart(
            px.bar(
                comparison_frame.dropna(subset=["Objective"]),
                x="Trial",
                y="Objective",
                color="Method",
                barmode="group",
                title="Illustrative objective by method and reproducible trial",
                hover_data=["Execution seconds", "Status"],
            ),
            use_container_width=True,
        )
        st.dataframe(comparison_frame, hide_index=True, use_container_width=True)
        if quantum:
            st.caption(
                "Trials use fixed seeds and identical constraints. A small simulator experiment "
                "does not establish quantum speedup or algorithmic superiority."
            )
            for index, result in enumerate(trial_results, start=1):
                with st.expander(f"QAOA trial {index}: {result['quantum_status']}"):
                    st.json(result["quantum_result"])
        else:
            st.info("Enable the local QAOA experiment to compare quantum-algorithm trials.")

        st.subheader("Optimization result")
        st.write(explain(solution, validation))
        if solution.get("quantum_result", {}).get("candidate_bitstring"):
            st.write(
                "Candidate bit string (candidate order A, B, C, D): "
                f"`{solution['quantum_result']['candidate_bitstring']}`"
            )
        explanations = explain_selection(solution["selected"], risk_weight, cost_weight)
        if explanations:
            st.subheader("Objective contribution of selected options")
            st.dataframe(pd.DataFrame(explanations), hide_index=True, use_container_width=True)
        else:
            st.write("No options were selected under the current objective and resource limit.")

        st.subheader("Independent feasibility checks")
        st.dataframe(
            pd.DataFrame([
                {"Check": name.replace("_", " ").title(), "Passed": passed}
                for name, passed in validation["checks"].items()
            ]),
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("QUBO representation")
        try:
            qubo = build_qubo(
                candidates,
                risk_weight=risk_weight,
                cost_weight=cost_weight,
                max_resource=max_resource,
            )
            st.caption(
                f"Variables: {qubo.get_num_vars()} · Linear constraints after conversion: "
                f"{len(qubo.linear_constraints)}"
            )
            st.code(qubo.prettyprint(), language="text")
        except (ImportError, ModuleNotFoundError) as exc:
            st.info(f"Install Qiskit Optimization to display the QUBO ({type(exc).__name__}).")

        st.subheader("Run audit")
        audit_record = {
            "run_number": len(st.session_state["run_history"]) + 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "synthetic_profile": profile,
            "excluded_candidate_ids": excluded_ids,
            "resource_limit": max_resource,
            "objective_weights": {"risk": risk_weight, "cost": cost_weight},
            "solver_requested": "QAOA" if quantum else "classical exact search",
            "qaoa_trials_requested": quantum_trials if quantum else 0,
            "solver_returned": solution["solver"],
            "quantum_status": (
                [result["quantum_status"] for result in trial_results]
                if quantum else "not requested"
            ),
            "qaoa_trial_configurations": [
                result["quantum_result"].get("configuration")
                for result in trial_results
            ],
            "classical_objective": classical["objective"],
            "final_objective": solution["objective"],
            "final_selected_ids": [item["id"] for item in solution["selected"]],
            "validation": validation,
        }
        st.session_state["run_history"].append(audit_record)
        st.json({
            "current_run": audit_record,
            "session_run_history": st.session_state["run_history"],
        })

        previous = st.session_state.get("previous_run")
        if previous:
            st.subheader("Synthetic feedback and re-optimization")
            st.caption(
                "The profile and optimizer are rerun from the submitted synthetic inputs. "
                "Option attributes are fictional and are not learned from outcomes."
            )
            previous_col, current_col = st.columns(2)
            previous_col.metric("Previous illustrative objective", f"{previous['objective']:.2f}")
            previous_col.write(f"Previous selected IDs: {previous['selected_ids'] or 'none'}")
            current_col.metric(
                "Current illustrative objective",
                f"{solution['objective']:.2f}",
                delta=f"{solution['objective'] - previous['objective']:+.2f}",
            )
            current_col.write(
                f"Current selected IDs: "
                f"{[item['id'] for item in solution['selected']] or 'none'}"
            )
            changed = {
                key: {"previous": previous["profile"][key], "current": value}
                for key, value in profile.items()
                if previous["profile"][key] != value
            }
            st.write("Changed synthetic profile values:", changed or "No profile values changed.")

        st.session_state["previous_run"] = {
            "profile": profile,
            "objective": solution["objective"],
            "selected_ids": [item["id"] for item in solution["selected"]],
            "probability": probability,
        }
    except ValueError as exc:
        st.error(str(exc))
else:
    st.info("Enter a synthetic profile in the sidebar and run the demonstration.")

with st.expander("How this prototype works"):
    st.markdown(
        "Profile validation → synthetic Random Forest trained against artificial labels → "
        "fictional option generation and explicit exclusions → exact classical search and, "
        "when requested, seeded QAOA using a local statevector simulator → independent "
        "eligibility, duplicate, and resource checks. Test metrics describe agreement with "
        "artificial labels only. The local simulator runs on a classical computer and does not "
        "demonstrate quantum hardware execution, clinical performance, or quantum speedup."
    )
